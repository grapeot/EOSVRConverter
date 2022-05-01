# EOS VR Utility Utility

This repo has two small utilities for Canon's great VR lens.
The `EOSVRConverter.py` basically does what EOS VR Utility does, with some bug fixes, capabilities to process RAW, and multi-core acceleration.
`enableJpgs.py` still uses Canon's EOS VR Utility, but makes it easier to use.

## EOS VR Converter

It converts an image or video from R5 + EOS VR lens to an equirectangular form, which could be rendered in VR goggles.
I used industry standard algorithms to develop this tool so the result might be different from EOS VR Utility's result.
Currently it doesn't depend on the EXIF data, so it doesn't require the image to be from the body, and thus could support jpg files derived from the RAW file.
However, this also limits its capability to perform automatic horizontal correction.
So there is no gaurantee the horizon is correct.
When the photo was taken with too much derivation from the horizonal position (tilt etc.) the photo may bring disorientation and dizziness.
The speed is about 10~20x faster than the EOS VR Utility though.
Similarly, the utility also supports RAW videos from R5 (indirectly), with multi-core acceleration.

### Usage

This is a python script, so one needs to have some basic understanding of python in order to use it.
First install python and dependencies in the requirement.
And then modify the code in the `EOSVRConverter.py` as you like, epsecially the `main` function.
It has quite some personalized adjustment embedded in, so I strongly suggest to first read the code instead of execute it blindly.

## EnableJpgs

When using the great EOS VR system, I got two pain points.
The first is the software doesn't accept RAW files, even for images.
We have to use the output jpg files directly from the body, instead of the more flexible and capable RAW files.
The second is the entire EOS VR Utility software is implemented with blocking behaviors.
That is, when clicking anything such as selecting a file, the UI will completely freeze while doing (usually time consuming) computation.
This is really annoying when one has tens or hundreds of photos to process.

This utility doesn't solve the two issues directly, but greatly alleviate the issues.
For the first one, it allows the EOS VR Utility to recognize the derived jpgs from RAW files.
And for the second one, it enabled "Horizontal Correction" by default, so we don't need to click into each image twice and wait for two minutes to click the "Hozirontal Correction" button.
We still need to click into image to calculate the parallax correction parameter though.

### Usage

This is a python script.
So please first install python 3, and also make sure `exittool` is installed and put into the system PATH.
Assuming the jpg files derived from the RAW files are stored in the folder `./jpgs`, and the original jpg files right from the camera are stored in the folder `./raw_jpgs`, then we can execute the following command:

```
python ./enableJpgs.py --jpgdir ./jpgs --rawdir ./raw_jpgs
```

This will write corresponding xml files to `./jpgs`, and patch the jpg files so the EOS VR Utility could recognize them.

## My Workflow

I also share my workflow of processing VR media here.

* After shooting using the (great) R5 and VR lens, I have a bunch of RAW files, and corresponding jpg files.
* I first use CaptureOne (or Lightroom) to pick the good pictures from the RAW files, edit them in the circular form, and export to high quality jpg in Adobe RGB color space (important, otherwise EOS VR Utility will output an over-saturated image).
* Then this tool is used to enable EOS VR Utility to recognize them.
* In the EOS VR Utility UI, a multiple selection by holding the Shift key and export would convert them to the square form.
* Then I usually use Topaz Gigapixel to enlarge them to 16k, which solves another pain point of insufficient resolution.
It sounds counter-intuitive because R5 is already an 8k camrea.
But the 8k is for the photos of both eyes.
So each eye only has 4k pixels horizontally.
And it needs to be further divided into 180 degrees.
So the resolution is actually not that high after alll the division.
It turned out such kind of artificial resolution boost is pretty effective on a Oculus Quest 2.