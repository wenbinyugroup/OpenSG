import numpy as np
from os.path import join, exists
import os
import yaml
from opensg.io.io import load_yaml


def generate_segment_shell_mesh_files(
    blade_mesh_yaml, segment_list=None, segment_folder=None
):
    """Generate segment mesh YAML files from a blade mesh YAML file.

    This function processes a blade mesh YAML file and creates individual YAML files
    for each blade segment. Each segment file contains all the necessary data to
    create a complete SegmentMesh object without requiring the full BladeMesh.

    Parameters
    ----------
    blade_mesh_yaml : str
        Path to the blade mesh YAML file
    segment_folder : str, optional
        Folder to save segment files in. If empty, creates a 'segments' folder.

    Returns
    -------
    list
        List of generated segment YAML file paths
    """

    # make folder if it doesn't exist
    if not segment_folder:
        segment_folder = "segments"
    if not os.path.exists(segment_folder):
        os.makedirs(segment_folder)

    # get name of blade mesh
    b_mesh_name = os.path.basename(blade_mesh_yaml)
    b_mesh_name = os.path.splitext(b_mesh_name)[0]

    b_mesh_data = load_yaml(blade_mesh_yaml)

    segment_mesh_data = dict()

    b_nodes = b_mesh_data["nodes"]
    num_b_nodes = len(b_nodes)

    b_elements = b_mesh_data["elements"]
    num_b_elements = len(b_elements)

    b_sets = b_mesh_data["sets"]
    b_materials = b_mesh_data["materials"]
    b_sections = b_mesh_data["sections"]
    b_element_orientations = b_mesh_data["elementOrientations"]

    # Generate material database
    # material_database = dict()

    # for i, material in enumerate(b_materials):
    #     material_dict = dict()

    #     material_dict["id"] = i

    #     elastic = material['elastic']
    #     material_dict["E"] = elastic['E']
    #     material_dict["G"] = elastic['G']
    #     material_dict["nu"] = elastic['nu']

    #     material_database[material['name']] = material_dict

    # Generate and save segment meshes
    if segment_list is None:
        segment_list = range(30)

    for segment_index in segment_list:
        filename = join(segment_folder, f"{b_mesh_name}_segment_{segment_index}.yaml")

        node_in_segment = np.zeros(
            num_b_nodes, dtype=int
        )  # Track which nodes are in the segment
        element_in_segment = np.zeros(
            num_b_elements, dtype=int
        )  # Track which elements are in the segment
        element_layer_id = np.zeros(
            num_b_elements, dtype=int
        )  # Track the layer number of each element

        # Convert blade section data to segment layup data
        mat_names, thick, angle = [], [], []
        for sec in b_sections:
            section_name_components = sec["elementSet"].split("_")
            if len(section_name_components) > 2:
                m, t, an = [], [], []
                if int(section_name_components[1]) == segment_index:
                    layup = sec["layup"]
                    for l in layup:
                        m.append(l[0])
                    mat_names.append(m)
                    for l in layup:
                        t.append(l[1])
                    thick.append(t)
                    for l in layup:
                        an.append(l[2])
                    angle.append(an)
        combined_rows = list(zip(thick, mat_names, angle))

        # Note: This assumes that the number of sets for a fixed segment is equal
        # to the number of sections for that segment. This is why ii can be used to
        # track both sets and sections.
        ii = 0
        unique_rows = []
        for es in b_sets["element"]:
            section_name_components = es["name"].split("_")
            if len(section_name_components) > 2:
                # Some section names do not have indices. These are assumed to represent
                # groups of multiple sections (eg all of the shear web elements)
                # and are ignored.
                try:
                    section_index = int(section_name_components[1])
                except ValueError:
                    continue

                if section_index == segment_index:
                    if combined_rows[ii] not in unique_rows:
                        unique_rows.append(combined_rows[ii])

                    lay_num = unique_rows.index(combined_rows[ii])
                    for eli in es["labels"]:
                        element_in_segment[eli] = 1
                        element_layer_id[eli] = lay_num
                        for nd in b_elements[eli]:
                            if nd > -1:
                                node_in_segment[nd] = 1
                    ii += 1

        # Check if segment has any elements
        if not np.any(element_in_segment):
            print(f"Segment {segment_index} has no elements, skipping...")
            continue

        # Re-extract layup data from unique_rows
        thick, mat_names, angle = (
            list(zip(*unique_rows))[0],
            list(zip(*unique_rows))[1],
            list(zip(*unique_rows))[2],
        )

        # Create mapping arrays for relabeling
        node_mapping = np.zeros(len(b_nodes), dtype=int)
        element_mapping = np.zeros(len(b_elements), dtype=int)

        # Build node mapping (blade index -> segment index starting at 1)
        segment_node_count = 1
        for i, is_in_segment in enumerate(node_in_segment):
            if is_in_segment:
                node_mapping[i] = segment_node_count
                segment_node_count += 1

        # Build element mapping (blade index -> segment index starting at 1)
        segment_element_count = 1
        for i, is_in_segment in enumerate(element_in_segment):
            if is_in_segment:
                element_mapping[i] = segment_element_count
                segment_element_count += 1

        # Extract segment-specific data
        segment_nodes = []
        segment_elements = []
        segment_orientations = []
        segment_subdomains = []

        # Extract nodes (in order of segment numbering)
        for i, nd in enumerate(b_nodes):
            if node_in_segment[i]:
                segment_nodes.append(nd)

        # Extract elements and orientations (in order of segment numbering)
        # Use the same logic as the blade segment: filter orientations based on element_in_segment
        segment_orientations = []
        for i, eo in enumerate(b_element_orientations):
            if element_in_segment[i]:  # Element is in segment
                o = []
                for k in range(9):
                    o.append(eo[k])
                segment_orientations.append(o)

        # Extract elements
        for i, el in enumerate(b_elements):
            if element_in_segment[i]:
                # Remap element node indices to local segment indices
                local_element = []
                for nd in el:
                    local_element.append(node_mapping[nd] - 1)  # Convert to 0-based

                segment_elements.append(local_element)

                # Extract subdomain (layer ID)
                segment_subdomains.append(element_layer_id[i])

        # Create segment YAML data
        segment_data = {
            "nodes": segment_nodes,
            "elements": segment_elements,
            "sets": {
                "element": [
                    {
                        "name": f"layup_{i}",
                        "labels": [
                            j
                            for j, subdomain in enumerate(segment_subdomains)
                            if subdomain == i
                        ],
                    }
                    for i in range(len(unique_rows))
                ]
            },
            "materials": b_materials,
            "sections": [
                {
                    "type": "shell",
                    "elementSet": f"layup_{i}",
                    "layup": [
                        [mat_names[i][j], thick[i][j], angle[i][j]]
                        for j in range(len(mat_names[i]))
                    ],
                }
                for i in range(len(unique_rows))
            ],
            "element_orientations": segment_orientations,
        }

        # Write segment YAML file using the same approach as blade YAML
        _mesh_to_segment_yaml(segment_data, filename)

        print(f"Generated segment {segment_index}: {filename}")

    return
    # return segment_mesh


def _mesh_to_segment_yaml(segment_mesh_data, file_name):
    """
    Convert segment mesh data to YAML format for standalone segment files.

    This function creates a YAML file for a segment using the same approach
    as the blade YAML creation, ensuring the segment is completely self-contained.

    Parameters
    ----------
    segment_mesh_data : dict
        Dictionary containing segment mesh data with keys:
        - nodes: list of node coordinates
        - elements: list of element connectivity
        - sets: dictionary of element sets
        - materials: list of material definitions
        - sections: list of section definitions
        - elementOrientations: list of element orientation matrices
    file_name : str
        Path to the output YAML file

    Returns
    -------
    None
    """
    mDataOut = dict()

    # Convert nodes to string format
    nodes = list()
    for nd in segment_mesh_data["nodes"]:
        ndstr = str(nd)
        nodes.append(ndstr)

    # Convert elements to string format
    elements = list()
    for el in segment_mesh_data["elements"]:
        elstr = str([int(e) for e in el])
        elements.append(elstr)

    # Convert element sets
    esList = list()
    for es in segment_mesh_data["sets"]["element"]:
        newSet = dict()
        newSet["name"] = es["name"]
        labels = list()
        for el in es["labels"]:
            labels.append(int(el))
        newSet["labels"] = labels
        esList.append(newSet)

    # Convert node sets (if they exist)
    # nsList = list()
    # try:
    #     for ns in segment_mesh_data["sets"]["node"]:
    #         newSet = dict()
    #         newSet["name"] = ns["name"]
    #         labels = list()
    #         for nd in ns["labels"]:
    #             labels.append(int(nd))
    #         newSet["labels"] = labels
    #         nsList.append(newSet)
    # except:
    #     pass

    # Convert sections
    sections = list()
    for sec in segment_mesh_data["sections"]:
        newSec = dict()
        newSec["type"] = sec["type"]
        newSec["elementSet"] = sec["elementSet"]
        if sec["type"] == "shell":
            newLayup = list()
            for lay in sec["layup"]:
                laystr = str(lay)
                newLayup.append(laystr)
            newSec["layup"] = newLayup
        else:
            newSec["material"] = sec["material"]
        sections.append(newSec)

    # Convert element orientations
    elOri = list()
    for ori in segment_mesh_data["element_orientations"]:
        elOri.append(str(ori))

    # Build output dictionary
    mDataOut["nodes"] = nodes
    mDataOut["elements"] = elements
    mDataOut["sets"] = dict()
    mDataOut["sets"]["element"] = esList
    # if nsList:
    #     mDataOut["sets"]["node"] = nsList
    mDataOut["sections"] = sections
    mDataOut["elementOrientations"] = elOri

    # Add materials if they exist
    try:
        mDataOut["materials"] = segment_mesh_data["materials"]
    except:
        pass

    # Write YAML file
    fileStr = yaml.dump(mDataOut, sort_keys=False)

    # Remove quotes
    fileStr = fileStr.replace("'", "")
    fileStr = fileStr.replace('"', "")

    with open(file_name, "w") as outStream:
        outStream.write(fileStr)
