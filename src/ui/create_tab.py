"""
Create tab UI component.
"""
import tkinter as tk
from tkinter import ttk


class CreateTab:
    """Create tab component."""

    def __init__(self, parent_notebook: ttk.Notebook):
        # Create the tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="4.Create")

        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets for the create tab."""
        # Main container
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(main_frame, text="Create", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # Placeholder content
        placeholder_label = ttk.Label(main_frame, text="Create functionality coming soon...",
                                    font=("Arial", 12), foreground="gray")
        placeholder_label.pack(pady=20)

        # You can add more widgets here in the future
        info_text = tk.Text(main_frame, height=10, width=60, wrap=tk.WORD, state=tk.DISABLED)
        info_text.pack(pady=10, fill="both", expand=True)

        # Add some placeholder text
        info_text.config(state=tk.NORMAL)
        info_text.insert(tk.END, "This tab will contain creation tools and features.\n\n")
        info_text.insert(tk.END, "Future features may include:\n")
        info_text.insert(tk.END, "• Character creation wizard\n")
        info_text.insert(tk.END, "• Image generation tools\n")
        info_text.insert(tk.END, "• Dataset creation utilities\n")
        info_text.insert(tk.END, "• Export/import functionality\n")
        info_text.config(state=tk.DISABLED)
