### HDR image databases:

All the HDR images are collected online

1. https://www2.cs.sfu.ca/~colour/data/funt_hdr/
2. http://rit-mcsl.org/fairchild//HDRPS/HDRthumbs.html
3. http://resources.mpi-inf.mpg.de/hdr/gallery.html
4. http://ivc.univ-nantes.fr/en/databases/ETHyma/
5. http://scarlet.stanford.edu/~brian/hdr/hdr.html
6. https://syns.soton.ac.uk/
7. http://pfstools.sourceforge.net/
8. https://ece.uwaterloo.ca/~z70wang/research/tmqi/
9. https://zenodo.org/record/1245790#.Xt6ugEVKhPY
10. https://github.com/AcademySoftwareFoundation/openexr-images
11. http://indoor.hdrdb.com/ 

Some images are in .exr format, and some are in .hdr format, we use FileStar to transform .exr format into .hdr format.

105 HDR images from 2 are used to test, other images are used to train **adTMO**.



### Target LDR images

All the collected HDR images are unlabeled, that is, we don't have the ground-truth LDR images. For each HDR image, we apply 30 TMOs in MATLAB (29 provided by HDR ToolBox and another is tonemap function) to get 30 LDR images, and select the one with highest TMQI as the ground-truth image, discard the other 29 LDR images.

We use this method the generate target 256 * 256 LDR images and target 1024 * 1024 LDR images (generating the target 1024 * 1024 LDR images will cost a lot of time).



### Resizing and cropping

For training, we use 256 * 256 resolution images, which come from two ways: 1. resizing original HDR images into 256 * 256 resolution; 2. resizing original HDR images into 1024 * 1024 resolution, and randomly cropping 256 * 256 resolution regions from it. 

![train](https://github.com/caoxingdong/adTMO/blob/master/databases/imgs/train.png?raw=true)

For testing, we use resized 256 * 256 resolution images and resized 1024* 2048 resolution images.