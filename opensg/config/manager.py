"""Configuration manager for OpenSG."""

import os
import yaml
from typing import Dict, Any, Optional
from .defaults import (
    SOLVER_SETTINGS,
    MESH_SETTINGS,
    DEFAULT_MATERIALS,
    ANALYSIS_SETTINGS,
    OUTPUT_SETTINGS
)


class ConfigManager:
    """Manages configuration settings for OpenSG.
    
    This class provides a centralized way to manage all configuration
    parameters, with support for loading from files and merging with
    default values.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the configuration manager.
        
        Parameters
        ----------
        config_file : str, optional
            Path to a YAML configuration file to load
        """
        self._config = {
            'solver': SOLVER_SETTINGS.copy(),
            'mesh': MESH_SETTINGS.copy(),
            'materials': DEFAULT_MATERIALS.copy(),
            'analysis': ANALYSIS_SETTINGS.copy(),
            'output': OUTPUT_SETTINGS.copy(),
        }
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from a YAML file.
        
        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file
        """
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Merge user configuration with defaults
        for section, settings in user_config.items():
            if section in self._config:
                self._config[section].update(settings)
            else:
                self._config[section] = settings
    
    def save_config(self, config_file: str) -> None:
        """Save current configuration to a YAML file.
        
        Parameters
        ----------
        config_file : str
            Path to save the configuration file
        """
        with open(config_file, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """Get configuration value(s).
        
        Parameters
        ----------
        section : str
            Configuration section name
        key : str, optional
            Specific key within the section. If None, returns entire section.
            
        Returns
        -------
        Any
            Configuration value(s)
        """
        if section not in self._config:
            raise KeyError(f"Configuration section '{section}' not found")
        
        if key is None:
            return self._config[section]
        
        if key not in self._config[section]:
            raise KeyError(f"Configuration key '{key}' not found in section '{section}'")
        
        return self._config[section][key]
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Parameters
        ----------
        section : str
            Configuration section name
        key : str
            Configuration key
        value : Any
            Value to set
        """
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section][key] = value
    
    def update(self, section: str, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values in a section.
        
        Parameters
        ----------
        section : str
            Configuration section name
        updates : dict
            Dictionary of key-value pairs to update
        """
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section].update(updates)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary.
        
        Returns
        -------
        dict
            Complete configuration dictionary
        """
        return self._config.copy() 