"""
Main application controller that coordinates all components.
"""
import warnings
# Suppress the pkg_resources deprecation warning from face_recognition_models
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys

# Add drag & drop support at application level
try:
    from tkinterdnd2 import TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    print("Warning: tkinterdnd2 not available. Drag & drop functionality will be disabled.")

# Import all our refactored components
from src.config.app_config import config
from src.models.character import CharacterRepository
from src.services.image_service import ImageService, StageProgressTracker
from src.services.gpu_service import get_gpu_service
from src.ui.character_tab import CharacterTab
from src.ui.images_tab import ImagesTab
from src.ui.provider_tab import ProviderTab
from src.ui.create_tab import CreateTab
from src.ui.splash_screen import SplashScreen
from src.ui.status_bar import StatusBar

# Import the processors and providers (these would need to be refactored too)
from src.processors import (
    ProviderImportProcessor,
    DuplicateFilterProcessor,
    UpscaleProcessor,
    SimpleResizeProcessor,
    InpaintingProcessor,
    QualityControlProcessor,
    FinalProcessor
)
from src.providers import ProviderManager


class DatasetAIManagerApp:
    """Main application controller."""

    def __init__(self):
        self.root = None
        self.splash = None

        # Initialize paths
        self.base_path = Path(__file__).parent
        self.characters_path = self.base_path / "characters"
        self.characters_path.mkdir(exist_ok=True)

        # Initialize GPU service
        self.gpu_service = get_gpu_service()

        # Initialize components (will be set during startup)
        self.character_repo = None
        self.image_service = None
        self.progress_tracker = None
        self.provider_manager = None
        self.processors = {}

        # Status bar for global status updates
        self.status_bar = None

        # Track GPU-dependent components for reinitialization
        self.gpu_dependent_components = []

    def show_splash_and_initialize(self):
        """Show splash screen and handle initialization."""
        # Initialize core services early for character pre-loading
        self._initialize_core_services()

        # Create and show splash screen with character services
        self.splash = SplashScreen(
            app_title=config.APP_NAME,
            version=config.APP_VERSION,
            character_repo=self.character_repo,
            image_service=self.image_service
        )
        self.splash.show()

        # Start initialization process
        self.splash.run_startup_sequence(
            on_complete_callback=self._on_startup_complete,
            on_error_callback=self._on_startup_error
        )

        # Keep splash screen running until initialization is complete
        self.splash.splash.mainloop()

    def _initialize_core_services(self):
        """Initialize core services needed for character pre-loading."""
        try:
            # Initialize character repository
            self.character_repo = CharacterRepository(self.characters_path)

            # Initialize image service first (without progress tracker)
            self.image_service = ImageService(self.characters_path)

            # Initialize progress tracker with the image service
            self.progress_tracker = StageProgressTracker(self.image_service)

            print("Core services initialized for character pre-loading")

        except Exception as e:
            print(f"Error initializing core services: {e}")
            # Set to None if initialization fails
            self.character_repo = None
            self.image_service = None
            self.progress_tracker = None

    def _on_startup_complete(self):
        """Called when startup sequence completes successfully."""
        # Schedule the transition to happen after splash screen mainloop exits
        def transition_to_main():
            # Close splash screen
            if self.splash and self.splash.splash:
                self.splash.splash.quit()  # Exit the mainloop
                self.splash.close()

            # Initialize the main application
            self._initialize_main_app()

        # Schedule the transition on the main thread
        if self.splash and self.splash.splash:
            self.splash.splash.after(100, transition_to_main)

    def _on_startup_error(self, error_message):
        """Called when startup sequence encounters an error."""
        # Close splash screen
        self.splash.close()

        # Show error message
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        messagebox.showerror("Startup Error", f"Failed to initialize application:\n{error_message}")
        root.destroy()
        sys.exit(1)

    def _initialize_main_app(self):
        """Initialize the main application after splash screen."""
        # Create main window with TkinterDnD support if available
        if DND_AVAILABLE:
            try:
                self.root = TkinterDnD.Tk()
                print("TkinterDnD initialization successful")
            except Exception as e:
                print(f"TkinterDnD initialization failed: {e}")
                self.root = tk.Tk()
                # Note: DND_AVAILABLE remains True globally, but TkinterDnD failed to initialize
        else:
            self.root = tk.Tk()

        self.root.title(f"{config.WINDOW_TITLE} v{config.APP_VERSION}")

        # Parse window dimensions from config
        # config.WINDOW_GEOMETRY is in format "WIDTHxHEIGHT"
        width_height = config.WINDOW_GEOMETRY.split('+')[0]  # Remove any existing position
        window_width, window_height = map(int, width_height.split('x'))

        # Calculate center position before showing window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Set geometry with center position in one go
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Set the application icon
        try:
            icon_path = self.base_path / "favicon.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
            else:
                print(f"Warning: Icon file not found at {icon_path}")
        except Exception as e:
            print(f"Warning: Could not set application icon: {e}")

        # Store reference to app in root for access by child components
        self.root.app = self

        # Initialize core services
        self._initialize_services()

        # Initialize processors
        self._initialize_processors()

        # Setup styles
        self._setup_styles()

        # Create UI
        self._create_ui()

        # Add status bar at the bottom with provider manager and characters path
        self.status_bar = StatusBar(self.root, provider_manager=self.provider_manager, characters_path=str(self.characters_path))
        self.status_bar.pack(side="bottom", fill="x")

        # Wire up component communication
        self._setup_component_communication()

    def set_status_bar_text(self, text):
        """Set the text of the status bar."""
        if self.status_bar:
            self.status_bar.set_status(text)

    def _initialize_services(self):
        """Initialize core business services."""
        self.character_repo = CharacterRepository(self.characters_path)
        self.image_service = ImageService(self.characters_path)
        self.progress_tracker = StageProgressTracker(self.image_service)
        self.provider_manager = ProviderManager()

    def _initialize_processors(self):
        """Initialize image processing components."""
        # Create a simple logging function for processors
        def log_function(message):
            print(message)

        def progress_function(progress):
            pass

        # Map processor keys to their corresponding classes
        processor_classes = {
            "0_to_1": ProviderImportProcessor,
            "1_to_2": DuplicateFilterProcessor,
            "2_to_3": UpscaleProcessor,
            "3_to_4": SimpleResizeProcessor,
            "4_to_5": InpaintingProcessor,
            "5_to_6": QualityControlProcessor,
            "6_to_7": FinalProcessor
        }

        # Initialize processors based on configuration
        self.processors = {}
        for processor_key in config.get_processor_keys_in_order():
            if processor_key in processor_classes:
                # Pass provider_manager to all processors
                self.processors[processor_key] = processor_classes[processor_key](
                    self.characters_path, log_function, progress_function, self.provider_manager
                )

    def _setup_styles(self):
        """Setup custom styles for the application."""
        style = ttk.Style()

        # Create a style for completed stages (green background)
        style.configure('Completed.TButton',
                       background='lightgreen',
                       foreground='darkgreen')

        # Create a style for selected completed stages
        style.map('Completed.TButton',
                 background=[('pressed', 'green'),
                           ('active', 'mediumseagreen')])

    def _create_ui(self):
        """Create the main UI components."""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create all tabs in numbered order
        self.character_tab = CharacterTab(self.notebook, self.character_repo, self.image_service)
        self.provider_tab = ProviderTab(self.notebook, self.provider_manager, self.characters_path)
        self.images_tab = ImagesTab(self.notebook, self.image_service, self.progress_tracker, self.character_repo, self)
        self.create_tab = CreateTab(self.notebook)

        # Pass pre-loaded character cache from splash screen to character tab
        if self.splash and hasattr(self.splash, 'character_info_cache'):
            self.character_tab.set_preloaded_character_cache(self.splash.character_info_cache)
            print(f"Transferred character cache from splash screen to character tab")

    def _setup_component_communication(self):
        """Wire up communication between components."""
        # Character tab callbacks
        self.character_tab.on_character_selected = self._on_character_selected
        self.character_tab.on_character_updated = self._on_character_updated

        # Images tab callbacks
        self.images_tab.on_face_image_set_requested = self._on_face_image_set_requested

        # GPU service callback
        self.gpu_service.add_selection_callback(self._notify_gpu_selection_changed)

    def _notify_gpu_selection_changed(self, device_info):
        """Handle GPU selection changes and notify components that need to reinitialize."""
        try:
            self.set_status_bar_text(f"GPU changed to: {device_info['name']}")

            # Notify processors that use GPU
            for processor_key, processor in self.processors.items():
                if hasattr(processor, 'reinitialize_with_selected_gpu'):
                    try:
                        processor.reinitialize_with_selected_gpu()
                        print(f"Reinitialized processor {processor_key} with new GPU")
                    except Exception as e:
                        print(f"Error reinitializing processor {processor_key}: {e}")

            # Notify other GPU-dependent components
            for component in self.gpu_dependent_components:
                if hasattr(component, 'reinitialize_with_selected_gpu'):
                    try:
                        component.reinitialize_with_selected_gpu()
                        print(f"Reinitialized component {type(component).__name__} with new GPU")
                    except Exception as e:
                        print(f"Error reinitializing component {type(component).__name__}: {e}")

        except Exception as e:
            print(f"Error handling GPU selection change: {e}")
            if self.status_bar:
                self.status_bar.set_status(f"Error changing GPU: {e}")

    def register_gpu_dependent_component(self, component):
        """Register a component that needs to be notified of GPU changes."""
        if component not in self.gpu_dependent_components:
            self.gpu_dependent_components.append(component)

    def _on_character_selected(self, character_name: str):
        """Handle character selection across all tabs."""
        # Update all tabs with the selected character
        self.images_tab.set_current_character(character_name)
        self.provider_tab.set_current_character(character_name)

        # Invalidate progress cache
        self.progress_tracker.invalidate_cache(character_name)

    def _on_character_updated(self):
        """Handle character updates."""
        # Refresh character list in character tab
        self.character_tab.refresh_character_list()

        # Update progress displays
        current_character = self.character_tab.get_current_character_name()
        if current_character:
            self.progress_tracker.invalidate_cache(current_character)
            self.images_tab._update_progress_display()

    def _process_single_stage_for_character_with_popup(self, character_name: str, processor_key: str, progress_popup):
        """Process a single stage for a character with progress popup updates."""
        if processor_key in self.processors:
            try:
                processor = self.processors[processor_key]

                # Store original functions - use correct attribute names
                original_log_callback = processor.log_callback
                original_progress_callback = processor.progress_callback

                # Redirect to progress popup
                processor.log_callback = progress_popup.log
                processor.progress_callback = progress_popup.update_stage_progress

                # Set progress dialog reference for processors that support UI integration
                if hasattr(processor, 'set_progress_dialog'):
                    processor.set_progress_dialog(progress_popup)

                try:
                    success = processor.process_character(character_name)
                finally:
                    # Restore original functions
                    processor.log_callback = original_log_callback
                    processor.progress_callback = original_progress_callback

                    # Clear progress dialog reference if it was set
                    if hasattr(processor, 'set_progress_dialog'):
                        processor.set_progress_dialog(None)

                # Invalidate progress cache to refresh displays
                self.progress_tracker.invalidate_cache(character_name)

                # Update progress displays
                self.images_tab._update_progress_display()

                return success
            except Exception as e:
                progress_popup.log_error(f"Error processing stage {processor_key}: {e}")
                print(f"Error processing stage {processor_key} for {character_name}: {e}")
                return False
        else:
            error_msg = f"Processor {processor_key} not found"
            progress_popup.log_error(error_msg)
            print(error_msg)
            return False

    def _on_face_image_set_requested(self, character_name: str, image_path: Path):
        """Handle face image setting requests."""
        from PIL import Image
        from pathlib import Path
        try:
            # Define destination path using Path objects
            dest_dir = Path("characters") / character_name
            dest_path = dest_dir / "face.png"
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Convert and copy image as PNG
            try:
                with Image.open(image_path) as img:
                    img.convert("RGBA").save(dest_path, format="PNG")
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to convert image to PNG: {e}")
                return

            # Load character data
            character = self.character_repo.load_character(character_name)
            if character:
                # Update face image path to full absolute path (as posix string)
                character.face_image = str(dest_path.resolve().as_posix())

                # Save character
                success = self.character_repo.save_character(character)
                if success:
                    # Update character tab display
                    self.character_tab._load_character_data(character_name)
                    # Update face image display in images tab
                    self.images_tab.refresh_face_image_display()
                else:
                    tk.messagebox.showerror("Error", "Failed to save face image")
            else:
                tk.messagebox.showerror("Error", "Character not found")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to set face image: {e}")

    def run(self):
        """Start the application."""
        # Add cleanup handler for proper shutdown
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _on_closing(self):
        """Handle application closing."""
        try:
            # Shutdown image service and cleanup resources
            self.image_service.shutdown()
            print("Image service shutdown complete")
            if self.status_bar:
                self.status_bar.stop()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            self.root.destroy()

def main():
    """Main entry point."""
    app = DatasetAIManagerApp()

    # Show splash screen and initialize
    app.show_splash_and_initialize()

    # If initialization was successful, run the main app
    if app.root:
        app.run()

if __name__ == "__main__":
    main()
