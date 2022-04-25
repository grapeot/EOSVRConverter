from glob import glob
from os.path import join, basename
from os import system
import argparse

# Note the horizontal correction is turned on by default.
XML_TEMPLATE = """<?xml version="1.0"?>
<VideoClip><FileVersion>1.0</FileVersion><Device><Manufacturer>Canon</Manufacturer><SerialNo>YOUR_CAMERA_SN</SerialNo><ModelName>Canon EOS R5</ModelName><FirmVersion>Firmware Version 1.5.0</FirmVersion><Lens>RF5.2mm F2.8 L DUAL FISHEYE</Lens></Device><Configuration><ParallaxCorrection><Status>ON</Status><Result>0</Result><Param1></Param1><Param2></Param2><Param3></Param3><Param4></Param4><Param5></Param5><Param6></Param6><Param7></Param7><Param8></Param8></ParallaxCorrection><HorizontalCorrection>ON</HorizontalCorrection><ManualHorizontalCorrection><Pan>0</Pan><Tilt>0</Tilt><Roll>0</Roll></ManualHorizontalCorrection></Configuration></VideoClip>"""

def enableJpgs(jpgDir, rawDir):
    jpgFns = glob(f'{jpgDir}/*.jpg')
    for fn in jpgFns:
        rawJpgFn = join(rawDir, basename(fn))
        cmd = f'exiftool -tagsfromfile {rawJpgFn} {fn}'
        system(cmd)
        xmlFn = fn + '.xml'
        open(xmlFn, 'w').write(XML_TEMPLATE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A small utility to allow Canon EOS VR Utility to indirectly process RAW files, and turn on horizontal correction by default.')
    parser.add_argument('--jpgdir', help='Directory holding the jpg files, which could be derived from RAW.', required=True)
    parser.add_argument('--rawdir', help='Directory holding the original jpg files from the camera body. We will copy the EXIF info from these files to the corresponding files.', required=True)
    args = parser.parse_args()    
    enableJpgs(args.jpgdir, args.rawdir)
