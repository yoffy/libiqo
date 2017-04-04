#!/bin/bash

set -e

rm -f out.yuv

FFMPEG=ffmpeg_g
#Method=iqo
#Method=cv_area
Method=cv_nearest
#Method=ipp_linear
#Method=ipp_cubic
#Method=ipp_super

# 537s
#Src=537s.yuv
#SrcW=1920
#SrcH=1080
#DstW=576
#DstH=324
#DstW=480
#DstH=270
#DstW=1366
#DstH=768
#DstW=1280
#DstH=720

# 537s720p
Src=537s720p.yuv
SrcW=1280
SrcH=720
DstW=800
DstH=450

# icon128
#Src=icon2014_128.yuv
#SrcW=128
#SrcH=128
#DstW=48
#DstH=48

# icon128 to 288
#Src=icon2014_128.yuv
#SrcW=128
#SrcH=128
#DstW=288
#DstH=288

# icon256
#Src=icon2014_256.yuv
#SrcW=256
#SrcH=256
#DstW=128
#DstH=128

# moire
#Src=moire622x756.yuv
#SrcW=622
#SrcH=756
#DstW=206
#DstH=250

# lena
#Src=lena512x512.yuv
#SrcW=512
#SrcH=512
#DstW=128
#DstH=128

#perf record --output=iqo_lanczos2.perf.data ./resize_yuv420p -d 1 -m ${Method} -i ${Src} -iw ${SrcW} -ih ${SrcH} -o out.yuv -ow ${DstW} -oh ${DstH}
#perf record --output=iqo_lanczos2_clang.perf.data ./resize_yuv420p -d 2 -i ${Src} -iw ${SrcW} -ih ${SrcH} -o out.yuv -ow ${DstW} -oh ${DstH}
#perf record --output=zimg_lanczos2.perf.data ./resize_yuv420p -d 2 -i ${Src} -iw ${SrcW} -ih ${SrcH} -o out.yuv -ow ${DstW} -oh ${DstH}
./resize_yuv420p -d 3 -m ${Method} -i ${Src} -iw ${SrcW} -ih ${SrcH} -o out.yuv -ow ${DstW} -oh ${DstH}
$FFMPEG -y -c:v rawvideo -s ${DstW}x${DstH} -pix_fmt yuv420p -i out.yuv ${Method}.png
#$FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags neighbor ff_near.png
#/bin/time -v $FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags bilinear ff_linear.png
#/bin/time -v $FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags bicubic ff_cubic.png
#/bin/time -v $FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags area ff_area.png
#$FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags lanczos ff_lanczos.png
#perf record --output=ff_lanczos2.perf.data $FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags lanczos -param0 2 ff_lanczos2.png
#$FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags lanczos -param0 3 ff_lanczos3.png
#$FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags lanczos -param0 4 ff_lanczos4.png
#$FFMPEG -y -c:v rawvideo -s ${SrcW}x${SrcH} -pix_fmt yuv420p -i ${Src} -c:v png -an -s ${DstW}x${DstH} -sws_flags lanczos -param0 10 ff_lanczos10.png

ffmpeg -y -i ${Method}.png -vf "crop=70:16:352:346" ${Method}_crop.png

#open ff_near.png
#open ff_lanczos.png
#open out.png
#open ff_lanczos.png icon2014_128.png out.png
#open ff_lanczos.png out.png
#eog ff_lanczos.png out.png &
eog ${Method}.png &

stty echo
