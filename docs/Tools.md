# Tools

Here is defined what are tools in the context of this project. And also the list of available tools.

## What are Tools?

Tools are defined in /src/tools. Each tool is a class that inherits from the base Tool class. Each tool has a name, description, and a run method. The run method is where the tool does its work. The run method takes input a single image and returns two images: processed_image, mask
Each tool is separated in two main run functions: detect, process. 
The detect function is used to detect if the tool can be applied to the image. For example, the GFPGAN tool will detect if there is a face in the image. If there is a face, it will return True and the bounding box of the face. The process function is used to apply the tool to the image. For example, the GFPGAN tool will take the bounding box of the face and apply the GFPGAN model to enhance the face.


## Available Tools

- **Background Removal**: Background removal tool using the RemoveBG API. It removes the background from the image and creates a mask for further processing.
- **Face Restoration**: Face restoration tool using the GFPGAN model. It detects faces in the image and enhances them for better quality, also creates mask.
- **U2Net**: Background removal tool using the U2Net model. It detects the main subject in the image and removes the background, creating a mask for further processing.
- **YOLO**: Object detection tool using the YOLO model. It detects various objects in the image and can be used for filtering or cropping based on detected objects, also creates mask.
- **Inpaint**: Image inpainting tool using Stable Diffusion. It fills in missing or masked areas of the image based on surrounding content. This one needs mask as input and input image.
- **Upscale**: Image upscaling tool using GFPGAN. It increases the resolution of the image for better quality.

## How can we access the tools in the UI?

In the Images Tab
We can see the tools when we right click on an image in the Images tab. We will see a context menu with the list of available tools. 
When we click on a tool, it will run the tool on the selected image. 
If the tool detects that it can be applied to the image, it will show a preview of the processed image and the mask.
We can then choose to apply the changes or cancel.

In the Images Viewer
When we are in the image viewer we can see all of the tools listed below as buttons. When we click on a button to execute a tool, it will run the tool on the current image.
And apply the upper requirements from Images Tab. When tool is run and mask is created we will show the mask in the image viewer as an overlay. We can also show multiple masks as multiple overlays. We can toggle the visibility of each mask overlay.
We will save the mask in RAM and also in the disk inside the character folder, so we can re-use it later if needed. The mask will be saved as a PNG file with the same name as the image and the tool name. For example, if the image is named "image1.jpg" and the tool is "GFPGAN", the mask will be saved as "image1_GFPGAN_mask.png". The directory within the character / {character_id} / tools / {tool_name} / {image_name}_{tool_name}_mask.png