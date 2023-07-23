# EOS VR Converter

This repo provides an alternative workflow than the EOS VR Utility for Canon's great VR lens.
The workflow aims to handle the current painpoints of the EOS VR Utility:

* RAW formats are unsupported for both still and video;
* Operation efficiency is not a focus of the software design, which has many blocking long operations. The users often have to wait for minutes before able to begin export;
* There are some artifacts in the parallex correction, sometimes resulting in dizziness or cross eyes.

The utility allows for the following workflows instead:

### Photos

* [Optional] Perform image adjustments from RAW files in your favorite editing tools. Export the result to JPG files.
* Use the `EOSVRConverter.py` to convert the images to equirectangular format in batch, in parallel, in a set and go manner.
* [Optional] If resolution is a concern, I usually use Topaz Sharpen AI to boost the clarity.

### Videos

* Perform RAW decoding, video color grading (if you use Canon Log) and other adjustment in your favorite editing tools. Export the result to MP4 files.
* If you use All-I format (instead of RAW formats), it's possible to skip the previous step and directly begin from the mp4 files right from the body. In this case, use the `EOSVRConverter.py` to do color grading, auto adjustment, and equirectangular transform to png files. It also extracts audio for future use. Otherwise, one can also use `EOSVRConverter.py` for equirectangular transform purpose only, similarly in a parallel, efficient, and set-and-go manner.
* Use `VideoCombiner.py` to optionally launch Topaz Sharpen AI to boost the clarity. And then the python script combines the frames and the audio into a MP4 file which can be played on VR goggles.

`EOSVRConverter_Red.py` provides support on RED V-Raptor VV. Check the code for more details.

`enableJpgs.py` is provided for legacy use only.

## Technical Details

This utility converts an image or video from R5 + EOS VR lens to an equirectangular form, which could be rendered in VR goggles.
I used industry standard algorithms to develop this tool so the result might be different from EOS VR Utility's result.
Currently it doesn't depend on the EXIF data, so it doesn't require the image to be from the body, and thus could support jpg files derived from the RAW file.
However, this also limits its capability to perform automatic horizontal correction.
So there is no gaurantee the horizon is correct.
When the photo was taken with too much derivation from the horizonal position (tilt etc.) the photo may bring disorientation and dizziness.
The speed is about 10~20x faster than the EOS VR Utility though.
Similarly, the utility also supports RAW videos from R5 (indirectly), with multi-core acceleration.

This is a python script, so one needs to have some basic understanding of python in order to use it.
First install python and dependencies in the requirement.
And then modify the code in the `EOSVRConverter.py` as you like, epsecially the `main` function.
It has quite some personalized adjustment embedded in, so I strongly suggest to first read the code instead of execute it blindly.

## [Legacy] EnableJpgs

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