This uses mss to take a screenshot of top right corner and opencv to analyze it
(only works with 5120x1440 monitor with normal colors e.g. no colorblind mode in overwatch.
These should be quite easy to change in code if needed). It searches for the read and blue
colors to find the "lines" next to killfeed icons. After finding those it uses template
matching to find which hero killed (or ressed) which. Finally, it uses template matching to 
find out what ability was used.

This data is then sent to dbot which can control how to react to different events.
