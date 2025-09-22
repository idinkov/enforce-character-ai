# Characters

## Overview

Characters are the main entities that are used to train models.

## Character Attributes

- `name`: The name of the character. (e.g., "John Doe")
- `aliases`: A list of alternative names or nicknames for the character. (e.g., ["Johnny", "JD"]) [optional]
- `description`: A brief description of the character. (e.g., "A brave warrior from the north.") [optional]
- `personality`: A description of the character's personality traits. (e.g., "Courageous, Loyal, Strategic") [optional]
- `hair_color`: The color of the character's hair. (e.g., "Black", "Blonde") [optional]
- `eye_color`: The color of the character's eyes. (e.g., "Brown", "Blue") [optional]
- `height_cm`: The height of the character (e.g., "176") [optional]
- `weight_kg`: The weight of the character (e.g., "70") [optional]
- `age`: The age of the character (e.g., "28") [optional]
- `birthdate`: The birthdate of the character (e.g., "1995-05-15") [optional]
- `gender`: The gender of the character (e.g. "Male", "Female") [optional]
- `image`: Path to an image representing the character. [optional]
- `face_image`: Path to an image representing the character's face. [optional]
- `training_prompt`: A text prompt used for training the character model. (e.g., "John Doe") [optional]

## Character Image Attributes
- `images.1_raw`: Array of images paths. First we gather raw images of the character from various sources.
- `images.2_raw_filtered`: Array of images paths. A list of paths to filtered raw images of the character. This one has deleted duplicates and images not containing the character.
- `images.3_raw_upscaled`: Array of images paths. A list of paths to upscaled images of the character. [optional]
- `images.4_processed_1024`: Array of images paths. A list of paths to processed images of the character. This one has been processed for training but not fully ready. It will contain images with multiple characters. Images will be cropped to 1024x1024 resolution.
- `images.5_processed_fixed_1024`: Array of images paths. This one contains folder with images from processed that are manually or automatically fixed (e.g., removing extra people, things, etc...)
- `images.6_rtt_1024`: Array of images paths. A list of paths to rtt images of the character.

## Characters Directory Structure

```
characters/
    ├── character_name/
    │   ├── character.yaml
    │   ├── images/
    │   │   ├── 1_raw/
    │   │   │   ├── image1.jpg
    │   │   │   ├── image2.png
    │   │   │   └── ...
    │   │   ├── 2_raw_filtered/
    │   │   │   ├── image1.jpg
    │   │   │   ├── image2.png
    │   │   │   └── ...
    │   │   ├── 3_raw_upscaled/
    │   │   │   ├── image1.jpg
    │   │   │   ├── image2.png
    │   │   │   └── ...
    
```

## Good Practices for Character Creation

- You need around 40-200 images of the character for a good training.
- Ensure diversity in the images (different angles, lighting, expressions, outfits).
- Do not include extra characters in the images.

## How are we creating dataset for character training

0. When character is created we are creating the folder structure and character.yaml file.
   1. Create folder `characters/character_name`
   2. Create folder `characters/character_name/images`
   3. Create subfolders `1_raw`, `2_raw_filtered`, `3_raw_upscaled`, `4_processed_1024`, `5_processed_fixed_1024`, `6_rtt_1024` inside `images` folder.
   4. Create `character.yaml` file inside `characters/character_name` folder.
   5. Add the character attributes to the `character.yaml` file.
   6. Add empty arrays for image attributes to the `character.yaml` file.

1. Gather raw images of the character from various sources.
   1. Download images from image search engines (e.g., Google Images, Bing Images).
   2. Scrape images from social media platforms (e.g., Instagram, Facebook).
   3. Manually select folder and we will upload them to 1_raw folder.
2. Filter out duplicates and images not containing the character.
   1. We can use tools like 
3. Upscale the filtered images to improve quality.
   1. You can use tools like Gigapixel AI or online upscaling services. [online]
   2. You can use offline tool like Real-ESRGAN to batch upscale images. [offline]
   3. You can use paid tool like Topaz Photo AI to batch upscale images. [offline-paid]
4. For the 4_processed phase we are going to use `face_image` to detect the character in the images and crop them to 1024x1024 resolution.
   1. We are gathering all of the photos from either `3_raw_upscaled` or `2_raw_filtered` or `1_raw` if the previous two do not exist.
   2. We are using `face_image` to detect the character in the images
   3. We are also detecting the number of faces in the image
   4. The hightest match of `face_image` with any face is our character
   5. If there are multiple faces we are going to see where they are in the photo.
   6. We are going to remove the background behind the characters from the photo. We are going to see if our face matches the character on the removed background. If there is character out of that area we are going to remove him. Then from the information of how to crop the photo crop the original photo to retain the background around our person.
   7. There may be need for inpainting if the aspect ratio is not 1:1
   8. Detect the zones that needs impaint and inpaint them using stable diffusion inpainting model.
   9. Save the cropped and inpainted images to `4_processed_1024` folder. With filenames `{$number_of_faces}_{$original_filename}_{has_inpaint}`.
5. We are going to go manually through the `4_processed_1024` images and fix them if needed. This may include removing extra characters, fixing cropping, etc...
   1. We are going to save the fixed images to `5_processed_fixed_1024` folder.

