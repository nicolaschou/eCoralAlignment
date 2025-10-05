from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from imageutils import load_image

class ReorderableListbox(ttk.Frame):
    """
    A scrollable listbox widget that allows drag-and-drop reordering.

    This class wraps a standard Tkinter Listbox inside a ttk.Frame.
    Users can click and drag items to change their order, which is kept
    synchronized with an underlying Python list.

    Attributes:
        lb (tk.Listbox): The underlying Tkinter listbox widget.
        scroll (ttk.Scrollbar): The vertical scrollbar.
        _items (list): The underlying Python list representing current
            items.
        _var (tk.Variable): String variable tracking listbox content.
        _drag_data (dict): Tracks drag state with keys "start_index" and
            "cur_index" for managing item movement during reordering.
    """

    def __init__(
            self,
            parent: tk.Widget,
            items: list | None = None,
            height=20
        ):
        """
        Initialize a ReorderableListbox object.

        Args:
            parent (tk.Widget): The parent container.
            items (list | None, optional): An initial list of items to
                populate the listbox. Each item is converted to a string
                for display. Defaults to `None`, which creates an empty
                list.
            height (int, optional): The number of visible rows in the
                listbox before scrolling is required. Defaults to 20.
        """
        ttk.Frame.__init__(self, parent, padding=6)
        
        self._items = items if items is not None else []
        self._var = tk.Variable(value=[str(x) for x in self._items])

        self.lb = tk.Listbox(
            self,
            listvariable=self._var,
            activestyle="none",
            selectmode=tk.SINGLE,
            height=height,
            exportselection=False,
        )
        self.lb.grid(row=0, column=0, sticky="nsew")

        self.scroll = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.lb.yview
        )
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.lb.configure(yscrollcommand=self.scroll.set)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Drag state
        self._drag_data = {
            "start_index": None,  # index where drag started
            "cur_index": None,  # current index during drag
        }

        # Action bindings
        self.lb.bind("<ButtonPress-1>", self._on_button_press)
        self.lb.bind("<B1-Motion>", self._on_mouse_drag)
        self.lb.bind("<ButtonRelease-1>", self._on_button_release)

    # Public -----------------------------------------------------------
    def set_items(self, items: list):
        """Set items."""
        self._items = list(items)
        self._var.set([str(x) for x in self._items])

    def get_items(self) -> list:
        """Return a copy of items."""
        return list(self._items)
    
    def append_items(self, items: list):
        """Add a list of items."""
        self._items.extend(items)
        self._var.set([str(x) for x in self._items])

        # Update selected and active item
        self.lb.selection_clear(0, tk.END)
        idx = self.lb.size() - 1
        self.lb.selection_set(idx)
        self.lb.activate(idx)
        self.lb.see(idx)
    
    def remove_selected(self) -> object | None:
        """Remove the currently selected item."""
        sel = self._current_selection_index()
        if sel is None:
            return None
        removed = self._items.pop(sel)
        self.lb.delete(sel)
        
        # Update selected and active item
        if self.lb.size():
            new_sel = min(sel, self.lb.size() - 1)
            self.lb.selection_set(new_sel)
            self.lb.activate(new_sel)
            self.lb.see(new_sel)
        self._var.set([str(x) for x in self._items])

        return removed
    
    # Internal ---------------------------------------------------------
    def _current_selection_index(self) -> int | None:
        """Return the index of the currently selected item."""
        try:
            sel = self.lb.curselection()
            if not sel:
                return None
            return int(sel[0])
        except Exception:
            return None

    def _index_nearest_y(self, y: int) -> int:
        """Return list index nearest to the given y coordinate."""
        if self.lb.size() == 0:
            return 0
        nearest = self.lb.nearest(y)
        return max(0, min(self.lb.size() - 1, nearest))

    def _move_item(self, i: int, j: int):
        """Move item from i to j in items."""
        if i == j:
            return
        # Update the list of objects
        item = self._items.pop(i)
        self._items.insert(j, item)
        
        # Update the listbox
        text = self.lb.get(i)
        self.lb.delete(i)
        self.lb.insert(j, text)

        # Update selected and active item
        self.lb.selection_clear(0, tk.END)
        self.lb.selection_set(j)
        self.lb.activate(j)
        self.lb.see(j)

    # Event Handling ---------------------------------------------------
    def _on_button_press(self, event: tk.Event):
        """Handle mouse button press to start a drag operation."""
        if self.lb.size() == 0:
            return
        
        # Update drag state
        idx = self._index_nearest_y(event.y)
        self._drag_data["start_index"] = idx
        self._drag_data["cur_index"] = idx

        # Update selected and active item
        self.lb.selection_clear(0, tk.END)
        self.lb.selection_set(idx)
        self.lb.activate(idx)
        self.lb.focus_set()

    def _on_mouse_drag(self, event: tk.Event):
        """Handle mouse drag events to reorder list items."""
        if self._drag_data["start_index"] is None:
            return
        
        # Update drag state
        new_index = self._index_nearest_y(event.y)
        cur_index = self._drag_data["cur_index"]
        if new_index != cur_index:
            # Move the item in listbox and in list of objects
            self._move_item(cur_index, new_index)
            self._drag_data["cur_index"] = new_index

    def _on_button_release(self, event: tk.Event):
        """Handle mouse button release to end a drag operation."""
        # Reset drag state
        if self._drag_data["start_index"] is None:
            return
        self._drag_data["start_index"] = None
        self._drag_data["cur_index"] = None


class AlignmentManager(ttk.Frame):
    """
    A GUI component for managing and organizing two lists of images:
    "Unaligned" and "Templates".

    This frame displays two side-by-side panels, each containing:
      - A title label ("Unaligned" or "Templates")
      - A ReorderableListbox for displaying images
      - Two control buttons: "Add Images" and "Remove Selected Image"

    Attributes:
        max_templates (int): The maximum number of template images.
        unaligned_list (ReorderableListbox): Listbox widget managing
            unaligned image items.
        templates_list (ReorderableListbox): Listbox widget managing
            template image items.
    """

    def __init__(self, parent: tk.Widget, max_templates: int = 10):
        """
        Initialize an AlignmentManager object.

        Args:
            parent (tk.Widget): The parent container.
            max_templates (int): The maximum number of template images.
        """
        ttk.Frame.__init__(self, parent, padding=10)

        self.max_templates = max_templates

        self.columnconfigure(0, weight=1, uniform="cols")
        self.columnconfigure(1, weight=1, uniform="cols")
        self.rowconfigure(0, weight=1)

        # Build both sides with one generic helper
        specs = [
            (0, "Unaligned", self._add_unaligned, self._remove_unaligned),
            (1, "Templates", self._add_templates, self._remove_templates),
        ]
        for column, title, add_cmd, remove_cmd in specs:
            pane, rl = self._build_panel(
                column=column,
                title=title,
                add_cmd=add_cmd,
                remove_cmd=remove_cmd
            )
            if title == "Unaligned":
                self.unaligned_list = rl
            else:
                self.templates_list = rl

    def _build_panel(
        self,
        column: int,
        title: str,
        add_cmd,
        remove_cmd,
    ):
        """
        Build a single side panel containing a title label, a
        reorderable list, and "Add" / "Remove" buttons.

        This helper method is used to generate one of the two panels
        ("Unaligned" or "Templates") in the AlignmentManager.

        Args:
            column (int): The column index in the parent grid where this
                panel will be placed (0 for left, 1 for right).
            title (str): Text label displayed at the top of the panel.
            add_cmd: Function to call when the "Add Images"
                button is pressed.
            remove_cmd: Function to call when the "Remove
                Selected Image" button is pressed.

        Returns:
            tuple[ttk.Frame, ReorderableListbox]: The panel and its
                ReorderableListbox.
        """
        pane = ttk.Frame(self)
        pane.grid(row=0, column=column, sticky="nsew")
        pane.columnconfigure(0, weight=1)
        pane.rowconfigure(1, weight=1)

        ttk.Label(
            pane,
            text=title,
            font=("TkDefaultFont", 24, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        rl = ReorderableListbox(pane)
        rl.grid(row=1, column=0, sticky="nsew")

        btns = ttk.Frame(pane)
        btns.grid(row=2, column=0, sticky="w", padx=(12, 0), pady=(4, 0))
        btns.columnconfigure((0, 1), weight=1)

        ttk.Button(
            btns,
            text="Add Images",
            command=add_cmd
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Button(
            btns,
            text="Remove Selected Image",
            command=remove_cmd
        ).grid(row=0, column=1, sticky="w")

        return pane, rl

    # Public -----------------------------------------------------------
    def get_unaligned_items(self) -> list:
        """Return the list of unaligned images."""
        return self.unaligned_list.get_items()

    def get_template_items(self) -> list:
        """Return the list of template images."""
        return self.templates_list.get_items()

    # Button Actions ---------------------------------------------------
    def _add_unaligned(self):
        """Add to the list of unaligned images."""
        imgs = get_images(self, "Select Unaligned Images")
        if imgs:
            self.unaligned_list.append_items(imgs)

    def _remove_unaligned(self):
        """Remove from the list of unaligned images."""
        self.unaligned_list.remove_selected()

    def _add_templates(self):
        """Add to the list of template images."""
        imgs = get_images(self, "Select Template Images")
        new_len = len(self.templates_list.get_items()) + len(imgs)
        if new_len > self.max_templates:
            messagebox.showerror(
                "Too Many Templates",
                f"Max: {self.max_templates}; New total would be {new_len}",
                parent=self,
            )
        else:
            self.templates_list.append_items(imgs)

    def _remove_templates(self):
        """Remove from the list of template images."""
        self.templates_list.remove_selected()


def get_images(parent: tk.Widget, heading: str) -> list:
    """
    Open a file dialog and return a list of the chosen images.

    Args:
        parent (tk.Widget): The parent container.
        heading (str): The text shown at the top of the file dialog.
    
    Returns:
        list: List of ImageData objects containing the selected images.
    """
    file_paths = filedialog.askopenfilenames(parent=parent, title=heading)
    images = []
    errors = []
    for path in file_paths:
        try:
            images.append(load_image(path))
        except Exception as e:
            errors.append(f"{Path(path).name}\n  {type(e).__name__}: {e}")

    if errors:
        # Show failures
        MAX = 8
        body = "\n\n".join(errors[:MAX])
        if len(errors) > MAX:
            body += f"\n\n…and {len(errors) - MAX} more."
        messagebox.showerror(
            "Failed to load some images",
            f"Could not load {len(errors)} file(s):\n\n{body}",
            parent=parent,
        )

    return images