* Useful because we use interactive data without having to use live interactions
* Changes q learning to only include in distribution actions (ie. actions that were actually seen in the dataset)
* Estimates the max using an upper expectile
* At inference does steps to ensure OOD actions are not chosen
*source: https://www.youtube.com/watch?v=fGq4np3brbs
