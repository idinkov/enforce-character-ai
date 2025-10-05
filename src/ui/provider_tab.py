"""
Provider management tab UI component.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional
import os
import datetime
import threading


class ProviderTab:
    """Provider management tab component."""

    def __init__(self, parent_notebook: ttk.Notebook, provider_manager, characters_path=None):
        self.provider_manager = provider_manager
        self.characters_path = characters_path or "characters"  # Default fallback
        self.current_character: Optional[str] = None
        self.auto_check_running = False

        # Create the tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="2.Providers")

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

        # Progress and log display
        self._create_progress_log_section()

    def _create_minimal_add_section(self):
        """Create minimal URL input and Add button section."""
        add_frame = ttk.LabelFrame(self.frame, text="Add Provider")
        add_frame.pack(fill="x", padx=10, pady=5)

        # URL input
        url_input_frame = ttk.Frame(add_frame)
        url_input_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(url_input_frame, text="URL or Folder:").pack(side="left")
        self.provider_url_var = tk.StringVar()
        self.provider_url_entry = ttk.Entry(url_input_frame, textvariable=self.provider_url_var, width=60)
        self.provider_url_entry.pack(side="left", fill="x", expand=True, padx=5)

        ttk.Button(url_input_frame, text="Browse", command=self._select_folder).pack(side="left", padx=2)
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
            self._log_message(f"Detecting provider for URL: {url}")
            self.frame.update()

            # Use provider manager to detect provider
            provider_info = self.provider_manager.detect_provider_from_url(url)
            if provider_info:
                provider_type = provider_info.get('type')
                params = provider_info.get('params', {})

                self.add_status_var.set(f"Detected {provider_type}, adding...")
                self._log_message(f"Detected {provider_type} provider, adding...")
                self.frame.update()

                # Construct full character directory path
                character_dir = os.path.join(self.characters_path, self.current_character)

                # Add provider with default options (download now and auto-check enabled)
                success = self.provider_manager.add_provider_to_character(
                    character_dir, provider_type, params,
                    download_now=True,
                    auto_check=True,
                    progress_callback=self._update_progress,
                    log_callback=self._log_message
                )

                if success:
                    self.add_status_label.config(foreground="green")
                    self.add_status_var.set(f"Successfully added {provider_type} provider")
                    self._log_message(f"Successfully added {provider_type} provider")
                    self._load_character_providers()
                    # Clear the URL field
                    self.provider_url_var.set("")
                else:
                    self.add_status_label.config(foreground="red")
                    self.add_status_var.set("Failed to add provider")
                    self._log_message("Failed to add provider")
            else:
                self.add_status_label.config(foreground="red")
                self.add_status_var.set("No provider detected for this URL/path")
                self._log_message("No provider detected for this URL/path")
        except Exception as e:
            self.add_status_label.config(foreground="red")
            self.add_status_var.set(f"Error: {str(e)}")
            self._log_message(f"Error adding provider: {str(e)}")
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
        """Check selected provider now using threaded operation to prevent UI freezing."""
        selection = self.providers_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a provider to check")
            return

        provider_index = selection[0]
        try:
            # Reset progress and start logging
            self._update_progress(0, 1, "Starting provider check...")
            self._log_message("Starting threaded provider check...")

            # Disable buttons during operation
            self.check_now_button.config(state="disabled")
            self.check_all_button.config(state="disabled")

            # Construct full character directory path
            character_dir = os.path.join(self.characters_path, self.current_character)
            providers = self.provider_manager.get_character_providers(character_dir)

            if provider_index >= len(providers):
                messagebox.showerror("Error", "Invalid provider selection")
                self._enable_buttons()
                return

            provider_config = providers[provider_index]
            provider_id = provider_config.get('id')
            provider_type = provider_config.get('type', 'unknown')

            if not provider_id:
                messagebox.showerror("Error", "Provider ID not found")
                self._enable_buttons()
                return

            self._log_message(f"Checking {provider_type} provider (ID: {provider_id})")

            def completion_callback(result):
                """Called when the threaded operation completes"""
                def ui_update():
                    if result >= 0:  # Success
                        self._update_progress(1, 1, "Complete")
                        if result > 0:
                            self._log_message(f"✓ Provider check completed successfully! {result} new files downloaded.")
                            messagebox.showinfo("Success", f"Provider check completed successfully!\n{result} new files downloaded.")
                        else:
                            self._log_message("✓ Provider check completed successfully! No new files found.")
                            messagebox.showinfo("Success", "Provider check completed successfully!\nNo new files found.")
                        # Refresh the provider list to show updated info
                        self._load_character_providers()
                    else:  # Error
                        self._update_progress(0, 1, "Failed")
                        self._log_message("✗ Provider check failed")
                        messagebox.showerror("Error", "Provider check failed")

                    # Re-enable buttons
                    self._enable_buttons()

                # Schedule UI update on main thread
                self.frame.after(0, ui_update)

            # Start threaded operation
            self.provider_manager.check_provider_now_threaded(
                character_dir, provider_id,
                progress_callback=self._update_progress,
                log_callback=self._log_message,
                completion_callback=completion_callback
            )

        except Exception as e:
            self._update_progress(0, 1, "Error")
            self._log_message(f"✗ Error checking provider: {str(e)}")
            messagebox.showerror("Error", f"Failed to check provider: {e}")
            self._enable_buttons()

    def _enable_buttons(self):
        """Re-enable provider operation buttons."""
        selection = self.providers_listbox.curselection()
        if selection:
            self.check_now_button.config(state="normal")
        self.check_all_button.config(state="normal")

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
        """Check all providers for the current character using threaded operations."""
        if not self.current_character:
            messagebox.showwarning("Warning", "Please select a character first")
            return

        try:
            # Reset progress and start logging
            self._update_progress(0, 1, "Starting provider checks...")
            self._log_message("Starting threaded check for all providers...")

            # Disable buttons during operation
            self.check_now_button.config(state="disabled")
            self.check_all_button.config(state="disabled")

            # Construct full character directory path
            character_dir = os.path.join(self.characters_path, self.current_character)
            providers = self.provider_manager.get_character_providers(character_dir)

            if not providers:
                self._log_message("No providers configured for this character")
                messagebox.showinfo("Info", "No providers configured for this character")
                self._enable_buttons()
                return

            def run_all_checks():
                """Run all provider checks in background thread"""
                try:
                    total_downloaded = 0
                    total_providers = len(providers)

                    self._log_message(f"Found {total_providers} provider(s) to check")

                    for i, provider_config in enumerate(providers):
                        provider_id = provider_config.get('id')
                        provider_type = provider_config.get('type', 'unknown')

                        if provider_id:
                            # Update progress for current provider
                            def update_progress():
                                self._update_progress(i, total_providers, f"Checking {provider_type}")
                            self.frame.after(0, update_progress)

                            def log_current():
                                self._log_message(f"Checking provider {i+1}/{total_providers}: {provider_type} (ID: {provider_id})")
                            self.frame.after(0, log_current)

                            downloaded_count = self.provider_manager.check_provider_now(
                                character_dir, provider_id,
                                progress_callback=self._threaded_progress_callback,
                                log_callback=self._threaded_log_callback
                            )

                            if downloaded_count >= 0:
                                total_downloaded += downloaded_count
                                if downloaded_count > 0:
                                    def log_success():
                                        self._log_message(f"✓ Downloaded {downloaded_count} new files from {provider_type}")
                                    self.frame.after(0, log_success)
                                else:
                                    def log_no_files():
                                        self._log_message(f"✓ No new files found from {provider_type}")
                                    self.frame.after(0, log_no_files)
                            else:
                                def log_failure():
                                    self._log_message(f"✗ Failed to check provider: {provider_type}")
                                self.frame.after(0, log_failure)

                    # Final completion on UI thread
                    def completion():
                        self._update_progress(total_providers, total_providers, "Complete")

                        if total_downloaded > 0:
                            self._log_message(f"✓ All providers checked! Total new files downloaded: {total_downloaded}")
                            messagebox.showinfo("Success", f"Checked all providers successfully! {total_downloaded} new files downloaded.")
                        else:
                            self._log_message("✓ All providers checked! No new files found.")
                            messagebox.showinfo("Success", "Checked all providers successfully! No new files found.")

                        # Refresh the provider list to show updated info
                        self._load_character_providers()
                        self._enable_buttons()

                    self.frame.after(0, completion)

                except Exception as e:
                    def error_handler():
                        self._update_progress(0, 1, "Error")
                        self._log_message(f"✗ Error checking providers: {str(e)}")
                        messagebox.showerror("Error", f"Failed to check providers: {e}")
                        self._enable_buttons()

                    self.frame.after(0, error_handler)

            # Start background thread
            thread = threading.Thread(target=run_all_checks, daemon=True)
            thread.start()

        except Exception as e:
            self._update_progress(0, 1, "Error")
            self._log_message(f"✗ Error checking providers: {str(e)}")
            messagebox.showerror("Error", f"Failed to check providers: {e}")
            self._enable_buttons()

    def _threaded_progress_callback(self, current, total, message=""):
        """Thread-safe progress callback that schedules UI updates on main thread"""
        def update():
            self._update_progress(current, total, message)
        self.frame.after(0, update)

    def _threaded_log_callback(self, message):
        """Thread-safe log callback that schedules UI updates on main thread"""
        def log():
            self._log_message(message)
        self.frame.after(0, log)

    def _create_progress_log_section(self):
        """Create the progress and log display section."""
        progress_log_frame = ttk.LabelFrame(self.frame, text="Provider Activity")
        progress_log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Progress bar
        progress_frame = ttk.Frame(progress_log_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(progress_frame, text="Progress:").pack(side="left")
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100, length=300)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5)

        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(side="right")

        # Log display
        log_frame = ttk.Frame(progress_log_frame)
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create text widget with scrollbar
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")

        # Clear log button
        ttk.Button(progress_log_frame, text="Clear Log", command=self._clear_log).pack(pady=2)

    def _clear_log(self):
        """Clear the log display."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _log_message(self, message: str):
        """Add a message to the log display."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)  # Auto-scroll to bottom
        self.log_text.config(state=tk.DISABLED)

        # Update UI immediately
        self.frame.update_idletasks()

    def _update_progress(self, current: int, total: int, message: str = ""):
        """Update the progress bar and label."""
        if total > 0:
            progress_percent = (current / total) * 100
            self.progress_var.set(progress_percent)

            if message:
                self.progress_label.config(text=f"{current}/{total} - {message}")
            else:
                self.progress_label.config(text=f"{current}/{total}")
        else:
            self.progress_var.set(0)
            self.progress_label.config(text=message or "Ready")

        # Update UI immediately
        self.frame.update_idletasks()

    def _select_folder(self):
        """Open folder dialog and automatically trigger add provider when folder is selected."""
        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            # Set the selected folder path in the entry field
            self.provider_url_var.set(folder_path)
            # Automatically trigger the add provider functionality
            self._add_provider_from_url()

