from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
        _var (tk.StringVar): String variable tracking listbox content.
        _drag_data (dict): Tracks drag state with keys "start_index" and
            "cur_index" for managing item movement during reordering.
    """

    def __init__(
        self,
        parent: tk.Widget,
        items: list | None = None,
        height: int = 20
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
        ttk.Frame.__init__(self, parent, padding=4)

        self._items = items if items is not None else []
        self._var = tk.StringVar(value=[str(x) for x in self._items])

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
        """Set the items for this listbox (replaces current items)."""
        self._items = list(items)
        self._var.set([str(x) for x in self._items])

    def get_items(self) -> list:
        """Return a shallow copy of items."""
        return list(self._items)

    def append_items(self, items: list):
        """Append a list of items to the current items."""
        if not items:
            return
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

    Each list is stored in a panel that contains:
      - A title label ("Unaligned" or "Templates").
      - A ReorderableListbox for displaying images.
      - Two control buttons: "Add Images" and "Remove Selected Image".

    The GUI also includes:
      - A button that prompts the user to select an output directory
      - A button that prompts the next step in alignment
      - A numeric entry box for the user to enter a number of keypoints
        when `_manual` is True.

    Attributes:
        max_templates (int): The maximum number of template images.
        unaligned_list (ReorderableListbox): Listbox widget managing
            unaligned image items.
        templates_list (ReorderableListbox): Listbox widget managing
            template image items.
        _out_dir (str | None): Path to the selected output directory,
            or None if no folder has been chosen yet.
        _out_dir_var (tk.StringVar): Backing variable for the footer
            label; reflects the selected output folder.
        _manual (bool): Whether this alignment manager is in manual
            mode, which adds an entry box in the footer for the number
            of keypoints and requires that the user selects a template.
            Defaults to False.
        _num_points_var (tk.StringVar): Created only when `_manual` is
            True. Holds the value of the number of keypoints in the
            footer entry box.
    """

    def __init__(
        self,
        parent: tk.Widget,
        max_templates: int,
        manual: bool = False
    ):
        """
        Initialize an AlignmentManager object.

        Args:
            parent (tk.Widget): The parent container.
            max_templates (int): The maximum number of template images.
            manual (bool, optional): Whether to enable manual mode,
                which adds an entry box in the footer for the number of
                keypoints and requires that the user selects a template.
                Defaults to False.
        """
        ttk.Frame.__init__(self, parent, padding=10)

        self.max_templates = max_templates
        self._out_dir = None
        self._out_dir_var = tk.StringVar(value="No folder selected")
        self._manual = manual

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

        # Footer containing "Select Output Folder" and "Done" buttons
        footer = ttk.Frame(self)
        footer.grid(row=1, column=0, columnspan=2,
                    sticky="ew", padx=(12, 12), pady=(16, 0))
        footer.columnconfigure(1, weight=1)

        ttk.Button(
            footer,
            text="Select Output Folder",
            command=self._choose_out_dir
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(footer, textvariable=self._out_dir_var).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        # Add number of keypoints entry if `_manual` is True
        if self._manual:
            # Register a validation callback that allows only digits
            vcmd = (self.register(self._validate_number), "%P")

            ttk.Label(footer, text="Number of Keypoints:").grid(
                row=0, column=2, sticky="e"
            )

            self._num_points_var = tk.StringVar(value="4")
            ttk.Entry(
                footer,
                textvariable=self._num_points_var,
                width=5,
                validate="key",
                validatecommand=vcmd
            ).grid(row=0, column=3, sticky="e", padx=(4, 16))

        ttk.Button(
            footer,
            text="Done",
            command=self._done
        ).grid(row=0, column=4, sticky="e")

    # Public -----------------------------------------------------------
    def get_unaligned_items(self) -> list:
        """Return the list of unaligned images."""
        return self.unaligned_list.get_items()

    def get_template_items(self) -> list:
        """Return the list of template images."""
        return self.templates_list.get_items()

    # Internal ---------------------------------------------------------
    def _build_panel(
        self,
        column: int,
        title: str,
        add_cmd,
        remove_cmd,
    ) -> tuple:
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
            tuple[ttk.Frame, ReorderableListbox]:
                - `panel` (ttk.Frame): The panel.
                - `rl` (ReorderableListbox): The reorderable list.
        """
        panel = ttk.Frame(self)
        panel.grid(row=0, column=column, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        ttk.Label(
            panel,
            text=title,
            font=("TkDefaultFont", 24, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        rl = ReorderableListbox(panel)
        rl.grid(row=1, column=0, sticky="nsew")

        btns = ttk.Frame(panel)
        btns.grid(row=2, column=0, sticky="w", padx=(12, 12), pady=(4, 0))
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
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))

        return panel, rl

    def _choose_out_dir(self):
        """Prompt user to select an output folder."""
        out_dir = filedialog.askdirectory(
            parent=self,
            title="Select Output Folder"
        )
        if out_dir:
            self._out_dir = out_dir
            self._out_dir_var.set(
                f"Output Folder: {Path(out_dir).name}"
            )

    def _validate_number(self, value: str) -> bool:
        """Allow only numeric or empty input."""
        return value.isdigit() or value == ""

    def _done(self):
        """
        Package results and close the window. The window is not closed
        if `_manual` is True and no template has been selected.
        """
        # Handle num_points depending on whether manual=True
        num_points = None
        if self._manual:
            # Ensure template has been selected
            if not self.get_template_items():
                messagebox.showerror(
                    "No Template Selected",
                    "A template is required for manual alignment.",
                    parent=self,
                )
                return
            value = self._num_points_var.get().strip()

            # Ensure number of keypoints is valid
            if value == "" or int(value) < 4:
                messagebox.showerror(
                    "Invalid Number of Keypoints",
                    "Number of keypoints must be at least 4.",
                    parent=self)
                return
            num_points = int(value)

        results = {
            "unaligned": self.get_unaligned_items(),
            "templates": self.get_template_items(),
            "out_dir": self._out_dir,
            "num_points": num_points
        }
        top = self.winfo_toplevel()
        setattr(top, "results", results)
        top.destroy()

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
        if not imgs:
            return
        new_len = len(self.templates_list.get_items()) + len(imgs)
        if new_len > self.max_templates:
            messagebox.showerror(
                "Too Many Templates",
                f"Max number of templates: {self.max_templates}; "
                f"New total would be {new_len}",
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
            "Failed To Load Some Images",
            f"Could not load {len(errors)} file(s):\n\n{body}",
            parent=parent,
        )

    return images
