"""
Character data models and management.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml
from src.config.app_config import AppConfig

# Get app config instance
_app_config = AppConfig()

@dataclass
class CharacterData:
    """Data model for a character."""
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    personality: str = ""
    hair_color: str = ""
    eye_color: str = ""
    height_cm: str = ""
    weight_kg: str = ""
    age: str = ""
    birthdate: str = ""
    gender: str = ""
    image: str = ""
    face_image: str = ""
    training_prompt: str = ""
    version: str = field(default_factory=lambda: _app_config.APP_VERSION)
    images: Dict[str, List[str]] = field(default_factory=dict)

    # Store any additional fields that might exist in YAML files
    extra_fields: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default images structure if not provided."""
        # Removed automatic creation of images structure as it's not being used
        pass

    @classmethod
    def from_dict(cls, data: dict) -> 'CharacterData':
        """Create CharacterData from dictionary, handling unknown fields gracefully."""
        # Get the known fields for CharacterData
        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(cls) if f.name != 'extra_fields'}

        # Separate known and unknown fields
        known_data = {}
        extra_data = {}

        for key, value in data.items():
            if key in known_fields:
                known_data[key] = value
            else:
                extra_data[key] = value

        # Create the instance with known fields
        instance = cls(**known_data)

        # Store any extra fields
        instance.extra_fields = extra_data

        return instance


class CharacterRepository:
    """Repository for character data persistence."""

    def __init__(self, characters_path: Path):
        self.characters_path = characters_path
        self.characters_path.mkdir(exist_ok=True)

    def get_all_character_names(self) -> List[str]:
        """Get list of all character names."""
        if not self.characters_path.exists():
            return []

        return [
            char_dir.name
            for char_dir in self.characters_path.iterdir()
            if char_dir.is_dir()
        ]

    def load_character(self, name: str) -> Optional[CharacterData]:
        """Load character data from file."""
        char_file = self.characters_path / name / "character.yaml"

        try:
            with open(char_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            # Ensure the name field is always present (use the directory name if missing)
            if 'name' not in data:
                data['name'] = name

            # Use the from_dict method to handle unknown fields gracefully
            return CharacterData.from_dict(data)
        except Exception as e:
            print(f"Error loading character {name}: {e}")
            return None

    def save_character(self, character: CharacterData) -> bool:
        """Save character data to file."""
        try:
            char_dir = self.characters_path / character.name
            char_dir.mkdir(exist_ok=True)

            char_file = char_dir / "character.yaml"

            # Ensure character has current version before saving
            if not character.version or character.version != _app_config.APP_VERSION:
                character.version = _app_config.APP_VERSION

            # Convert dataclass to dict for YAML serialization
            data = {
				'version': character.version,
                'name': character.name,
                'aliases': character.aliases,
                'description': character.description,
                'personality': character.personality,
                'hair_color': character.hair_color,
                'eye_color': character.eye_color,
                'height_cm': character.height_cm,
                'weight_kg': character.weight_kg,
                'age': character.age,
                'birthdate': character.birthdate,
                'gender': character.gender,
                'image': character.image,
                'face_image': character.face_image,
                'training_prompt': character.training_prompt,
            }

            with open(char_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            return True
        except Exception as e:
            print(f"Error saving character {character.name}: {e}")
            return False

    def delete_character(self, name: str) -> bool:
        """Delete character and all its data."""
        try:
            import shutil
            char_path = self.characters_path / name

            if char_path.exists():
                shutil.rmtree(char_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting character {name}: {e}")
            return False

    def create_character_structure(self, name: str, stage_folders: List[str]) -> bool:
        """Create directory structure for a new character."""
        try:
            char_path = self.characters_path / name
            char_path.mkdir(exist_ok=True)

            # Create images directory structure
            images_path = char_path / "images"
            images_path.mkdir(exist_ok=True)

            for folder in stage_folders:
                (images_path / folder).mkdir(exist_ok=True)

            # Create and save initial character data
            character = CharacterData(name=name)
            self.save_character(character)

            return True
        except Exception as e:
            print(f"Error creating character structure for {name}: {e}")
            return False
