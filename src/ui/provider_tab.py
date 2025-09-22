"""
Provider management tab UI component.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
import os


class ProviderTab:
    """Provider management tab component."""

    def __init__(self, parent_notebook: ttk.Notebook, provider_manager, characters_path=None):
        self.provider_manager = provider_manager
        self.characters_path = characters_path or "characters"  # Default fallback
        self.current_character: Optional[str] = None
        self.auto_check_running = False

        # Create the tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="Providers")

        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets for the provider tab."""
        # Character selection display
        char_select_frame = ttk.Frame(self.frame)
        char_select_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(char_select_frame, text="Selected Character:").pack(side="left")
        self.provider_selected_char_label = ttk.Label(char_select_frame, text="None", font=("Arial", 10, "bold"))
        self.provider_selected_char_label.pack(side="left", padx=10)

        # Minimal URL input and Add section
        self._create_minimal_add_section()

        # Current providers list
        self._create_providers_list_section()

    def _create_minimal_add_section(self):
        """Create minimal URL input and Add button section."""
        add_frame = ttk.LabelFrame(self.frame, text="Add Provider")
        add_frame.pack(fill="x", padx=10, pady=5)

        # URL input
        url_input_frame = ttk.Frame(add_frame)
        url_input_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(url_input_frame, text="URL:").pack(side="left")
        self.provider_url_var = tk.StringVar()
        self.provider_url_entry = ttk.Entry(url_input_frame, textvariable=self.provider_url_var, width=60)
        self.provider_url_entry.pack(side="left", fill="x", expand=True, padx=5)

        ttk.Button(url_input_frame, text="Add", command=self._add_provider_from_url).pack(side="left", padx=5)

        # Status/result display
        self.add_status_var = tk.StringVar(value="Enter a URL or folder path and click Add")
        self.add_status_label = ttk.Label(add_frame, textvariable=self.add_status_var,
                                         font=("Arial", 9), foreground="gray")
        self.add_status_label.pack(pady=2)

    def _create_providers_list_section(self):
        """Create the providers list section."""
        providers_frame = ttk.LabelFrame(self.frame, text="Current Providers")
        providers_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Providers listbox with scrollbar
        providers_list_frame = ttk.Frame(providers_frame)
        providers_list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.providers_listbox = tk.Listbox(providers_list_frame)
        providers_scrollbar = ttk.Scrollbar(providers_list_frame, orient="vertical", command=self.providers_listbox.yview)
        self.providers_listbox.config(yscrollcommand=providers_scrollbar.set)

        self.providers_listbox.pack(side="left", fill="both", expand=True)
        providers_scrollbar.pack(side="right", fill="y")

        # Bind selection event to update button states
        self.providers_listbox.bind('<<ListboxSelect>>', self._on_provider_selection_changed)

        # Provider action buttons
        provider_actions_frame = ttk.Frame(providers_frame)
        provider_actions_frame.pack(fill="x", padx=5, pady=5)

        self.check_all_button = ttk.Button(provider_actions_frame, text="Check All", command=self._check_all_providers)
        self.check_all_button.pack(side="left", padx=2)

        self.check_now_button = ttk.Button(provider_actions_frame, text="Check Now", command=self._check_provider_now, state="disabled")
        self.check_now_button.pack(side="left", padx=2)

        ttk.Button(provider_actions_frame, text="Remove", command=self._remove_provider).pack(side="left", padx=2)
        ttk.Button(provider_actions_frame, text="Refresh", command=self._load_character_providers).pack(side="left", padx=2)

    def set_current_character(self, character_name: Optional[str]):
        """Set the current character."""
        self.current_character = character_name
        self.provider_selected_char_label.config(text=character_name or "None")
        self._load_character_providers()

    def _add_provider_from_url(self):
        """Auto-detect provider from URL and add it automatically."""
        if not self.current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        url = self.provider_url_var.get().strip()
        if not url:
            messagebox.showwarning("Warning", "Please enter a URL or folder path")
            return

        try:
            self.add_status_label.config(foreground="blue")
            self.add_status_var.set("Detecting provider...")
            self.frame.update()

            # Use provider manager to detect provider
            provider_info = self.provider_manager.detect_provider_from_url(url)
            if provider_info:
                provider_type = provider_info.get('type')
                params = provider_info.get('params', {})

                self.add_status_var.set(f"Detected {provider_type}, adding...")
                self.frame.update()

                # Construct full character directory path
                character_dir = os.path.join(self.characters_path, self.current_character)

                # Add provider with default options (download now and auto-check enabled)
                success = self.provider_manager.add_provider_to_character(
                    character_dir, provider_type, params,
                    download_now=True,
                    auto_check=True
                )

                if success:
                    self.add_status_label.config(foreground="green")
                    self.add_status_var.set(f"Successfully added {provider_type} provider")
                    self._load_character_providers()
                    # Clear the URL field
                    self.provider_url_var.set("")
                else:
                    self.add_status_label.config(foreground="red")
                    self.add_status_var.set("Failed to add provider")
            else:
                self.add_status_label.config(foreground="red")
                self.add_status_var.set("No provider detected for this URL/path")
        except Exception as e:
            self.add_status_label.config(foreground="red")
            self.add_status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to add provider: {e}")

    def _load_character_providers(self):
        """Load providers for current character."""
        self.providers_listbox.delete(0, tk.END)

        if not self.current_character:
            return

        try:
            # Construct full character directory path
            character_dir = os.path.join(self.characters_path, self.current_character)
            providers = self.provider_manager.get_character_providers(character_dir)
            for provider in providers:
                # Use the display name method
                display_text = self.provider_manager.get_provider_display_name(provider)
                self.providers_listbox.insert(tk.END, display_text)
        except Exception as e:
            print(f"Error loading providers: {e}")

    def _check_provider_now(self):
        """Check selected provider now."""
        selection = self.providers_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a provider to check")
            return

        provider_index = selection[0]
        try:
            # Construct full character directory path
            character_dir = os.path.join(self.characters_path, self.current_character)
            providers = self.provider_manager.get_character_providers(character_dir)

            if provider_index >= len(providers):
                messagebox.showerror("Error", "Invalid provider selection")
                return

            provider_config = providers[provider_index]
            provider_id = provider_config.get('id')

            if not provider_id:
                messagebox.showerror("Error", "Provider ID not found")
                return

            downloaded_count = self.provider_manager.check_provider_now(character_dir, provider_id)

            # check_provider_now returns number of files downloaded, not boolean success
            if downloaded_count >= 0:  # Any non-negative number means success
                if downloaded_count > 0:
                    messagebox.showinfo("Success", f"Provider check completed successfully!\n{downloaded_count} new files downloaded.")
                else:
                    messagebox.showinfo("Success", "Provider check completed successfully!\nNo new files found.")
                # Refresh the provider list to show updated info
                self._load_character_providers()
            else:
                messagebox.showerror("Error", "Provider check failed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check provider: {e}")

    def _remove_provider(self):
        """Remove selected provider."""
        selection = self.providers_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a provider to remove")
            return

        if messagebox.askyesno("Confirm", "Are you sure you want to remove this provider?"):
            provider_index = selection[0]
            try:
                # Construct full character directory path
                character_dir = os.path.join(self.characters_path, self.current_character)
                providers = self.provider_manager.get_character_providers(character_dir)

                if provider_index >= len(providers):
                    messagebox.showerror("Error", "Invalid provider selection")
                    return

                provider_config = providers[provider_index]
                provider_id = provider_config.get('id')

                if not provider_id:
                    messagebox.showerror("Error", "Provider ID not found")
                    return

                success = self.provider_manager.remove_provider_from_character(character_dir, provider_id)
                if success:
                    self._load_character_providers()
                else:
                    messagebox.showerror("Error", "Failed to remove provider")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to remove provider: {e}")

    def _on_provider_selection_changed(self, event):
        """Handle provider selection change."""
        selection = self.providers_listbox.curselection()
        # Enable or disable buttons based on selection
        if selection:
            self.check_now_button.config(state="normal")
        else:
            self.check_now_button.config(state="disabled")

    def _check_all_providers(self):
        """Check all providers for the current character."""
        if not self.current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        try:
            # Construct full character directory path
            character_dir = os.path.join(self.characters_path, self.current_character)
            providers = self.provider_manager.get_character_providers(character_dir)

            total_downloaded = 0
            for provider_config in providers:
                provider_id = provider_config.get('id')
                if provider_id:
                    downloaded_count = self.provider_manager.check_provider_now(character_dir, provider_id)
                    if downloaded_count >= 0:
                        total_downloaded += downloaded_count

            if total_downloaded > 0:
                messagebox.showinfo("Success", f"Checked all providers successfully! {total_downloaded} new files downloaded.")
            else:
                messagebox.showinfo("Success", "Checked all providers successfully! No new files found.")

            # Refresh the provider list to show updated info
            self._load_character_providers()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check providers: {e}")
