/* Click-to-enlarge for the pictures of the CompositesAI tutorials (figure.chat-figure, written by
 * _tools/chat_to_tutorial.py).
 *
 * Sphinx links each picture to its full-size file (html_scaled_image_link). A plain click opens that file in an
 * overlay on the page instead, so the reader stays on the tutorial. A click anywhere on the overlay, its close button
 * or the Escape key closes it. A click with a modifier key (a new tab or window) and a page without JavaScript keep
 * the plain link. The overlay is styled in chat.css (.chat-zoom).
 */
(function () {
  "use strict";

  var overlay = null;
  var picture = null;
  var closeButton = null;
  var opener = null;

  function close() {
    if (!overlay || overlay.hidden) {
      return;
    }
    overlay.hidden = true;
    picture.removeAttribute("src");
    document.documentElement.classList.remove("chat-zoom-open");
    if (opener) {
      opener.focus();
      opener = null;
    }
  }

  function build() {
    overlay = document.createElement("div");
    overlay.className = "chat-zoom";
    overlay.hidden = true;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Enlarged picture");

    closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.className = "chat-zoom-close";
    closeButton.setAttribute("aria-label", "Close the enlarged picture");
    closeButton.textContent = "×";

    picture = document.createElement("img");
    picture.className = "chat-zoom-image";

    overlay.appendChild(closeButton);
    overlay.appendChild(picture);
    overlay.addEventListener("click", close);
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        close();
      }
    });
    document.body.appendChild(overlay);
  }

  function open(link) {
    if (!overlay) {
      build();
    }
    var shown = link.querySelector("img");
    opener = link;
    picture.src = link.href;
    picture.alt = shown ? shown.alt : "";
    overlay.hidden = false;
    document.documentElement.classList.add("chat-zoom-open");
    closeButton.focus();
  }

  document.addEventListener("click", function (event) {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey ||
        event.altKey || !event.target.closest) {
      return;
    }
    var link = event.target.closest("figure.chat-figure a.image-reference");
    if (!link) {
      return;
    }
    event.preventDefault();
    open(link);
  });
})();
