# TODO

Fixes:
-> Improve provider to support progress tracking of the import. Implement in stage 0 to 1.
-> Fix YoloV8 to run on GPU and if GPU is not available fallback to CPU. Record this so in the current session we don't try to use GPU again if it failed once.
-> Add hide to tray option. And right click menu on tray icon to open app, open models dir, exit app, check providers, check for updates.
-> When Automatic GFPGAN is enabled, it upscales only the faces.
-> When cropping face image, rotate the face to be straight, and also crop around 20% more area, like the entire head.

Introduce comprehensive versioning system:
-> Implement semantic versioning (e.g., MAJOR.MINOR.PATCH) to track changes and updates to the application.
-> Maintain a changelog to document new features, bug fixes, and improvements for each version release.
-> Ensure that all dependencies and libraries used in the project are also versioned and compatible with the main application version.
-> Make Character Migrator. Ensure migrations between versions are smooth and do not disrupt user experience.

Make relationships between images and stages:
-> On each stage we should be able to track the lifecycle of an image. For example if we are on stage 0 providers import and we already have all the stages completed, we should be able to see in the image which stage its on. And if it has been deleted on a later stage. This will allow us to track the lifecycle of an image and avoid re-adding images that have been deleted on later stages.
-> Image Viewer will be able to see the image between stages and see the history of the image.
-> Images Tab will be able to filter images within certain selected stage, how far they have gone in the pipeline, and sort by that.
-> In Images Tab you will be easily able to see which images didnt go to next stage and why (e.g., failed quality checks, deleted by user, automatic filtering, etc.)

Create Archives:
-> Allow users to create ZIP or TAR.GZ archives of characters. It will include all of the character files inside the character folder.

To Reasearch:
-> https://github.com/DavidDinkevich/Story2Board 

Big TODOs:
-> Create API
-> Create CLI
-> Create Web UI
-> Create MCP Server so this can be used in LLMs 
-> Create Dockerfile and docker-compose.yml
-> Create Tests
-> Create Documentation
-> Create Installers for Windows, MacOS, Linux
-> Create Auto Updater
-> Publish on DockerHub
-> Publish on GitHub