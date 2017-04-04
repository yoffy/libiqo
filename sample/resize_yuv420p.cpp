#include <stdint.h>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include <libiqo/IQOLanczosResizer.hpp>
#include <opencv2/imgproc.hpp>
#include <ipp.h>

namespace {

    std::map<std::string, std::string> getArgs(int argc, char * argv[])
    {
        std::map<std::string, std::string> args;
        int i = 0;

        while ( i < argc ) {
            if ( argv[i][0] == '-' && i + 1 < argc ) {
                args[argv[i] + 1] = argv[i + 1];
                i += 2;
            } else {
                args[argv[i]] = "true";
                ++i;
            }
        }

        return args;
    }

    int64_t now_ns()
    {
        timespec spec;
        clock_gettime(CLOCK_REALTIME, &spec);
        return spec.tv_sec * 1000*1000*1000 + spec.tv_nsec;
    }

    void ResizeIQO(
        int degree,
        intptr_t srcStX,
        intptr_t srcW,
        intptr_t srcStY,
        intptr_t srcH,
        const uint8_t * srcY,
        const uint8_t * srcU,
        const uint8_t * srcV,
        intptr_t dstStX,
        intptr_t dstW,
        intptr_t dstStY,
        intptr_t dstH,
        uint8_t * dstY,
        uint8_t * dstU,
        uint8_t * dstV)
    {
        // resize Y
        {
            iqo::LanczosResizer r(degree, srcW, srcH, dstW, dstH);

            r.resize(srcStX, srcY, dstStX, dstY);
        }
        // resize U and V
        {
            iqo::LanczosResizer r(degree, srcStX / 2, srcStY / 2, dstStX / 2, dstStY / 2, 2);

            r.resize(srcStX / 2, srcU, dstStX / 2, dstU);
            r.resize(srcStX / 2, srcV, dstStX / 2, dstV);
        }
    }

    void ResizeCV(
        int degree,
        intptr_t srcStX,
        intptr_t srcW,
        intptr_t srcStY,
        intptr_t srcH,
        const uint8_t * srcY,
        const uint8_t * srcU,
        const uint8_t * srcV,
        intptr_t dstStX,
        intptr_t dstW,
        intptr_t dstStY,
        intptr_t dstH,
        uint8_t * dstY,
        uint8_t * dstU,
        uint8_t * dstV,
        int algorithm)
    {
        (void)degree;
        (void)srcStX;
        (void)srcStY;
        (void)dstStX;
        (void)dstStY;

        // resize Y
        {
            int srcSizes[] = { int(srcH), int(srcW) };
            int dstSizes[] = { int(dstH), int(dstW) };
            cv::Mat imgSrcY(2, srcSizes, CV_8UC1, const_cast<uint8_t*>(srcY));
            cv::Mat imgDstY(2, dstSizes, CV_8UC1, const_cast<uint8_t*>(dstY));
            cv::resize(imgSrcY, imgDstY, imgDstY.size(), 0, 0, algorithm);
        }
        // resize U and V
        {
            int srcSizes[] = { int(srcH / 2), int(srcW / 2) };
            int dstSizes[] = { int(dstH / 2), int(dstW / 2) };
            cv::Mat imgSrcU(2, srcSizes, CV_8UC1, const_cast<uint8_t*>(srcU));
            cv::Mat imgSrcV(2, srcSizes, CV_8UC1, const_cast<uint8_t*>(srcV));
            cv::Mat imgDstU(2, dstSizes, CV_8UC1, const_cast<uint8_t*>(dstU));
            cv::Mat imgDstV(2, dstSizes, CV_8UC1, const_cast<uint8_t*>(dstV));
            cv::resize(imgSrcU, imgDstU, imgDstU.size(), 0, 0, algorithm);
            cv::resize(imgSrcV, imgDstV, imgDstV.size(), 0, 0, algorithm);
        }
    }

    void ResizeIPPLinear(
        int degree,
        intptr_t srcStX,
        intptr_t srcW,
        intptr_t srcStY,
        intptr_t srcH,
        const uint8_t * srcY,
        const uint8_t * srcU,
        const uint8_t * srcV,
        intptr_t dstStX,
        intptr_t dstW,
        intptr_t dstStY,
        intptr_t dstH,
        uint8_t * dstY,
        uint8_t * dstU,
        uint8_t * dstV)
    {
        (void)degree;
        (void)srcStY;
        (void)dstStY;

        // resize Y
        {
            IppiSize srcSize = {int(srcW), int(srcH)};
            IppiSize dstSize = {int(dstW), int(dstH)};
            IppiPoint offset = {0, 0};
            int specSize = 0;
            int initBufSize = 0;
            int bufSize = 0;
            IppStatus status;
            status = ippiResizeGetSize_8u(srcSize, dstSize, ippLinear, 0, &specSize, &initBufSize);
            if ( status ) {
                std::printf("ippiResizeGetSize_8u(Y): %d\n", status);
                return;
            }
            IppiResizeSpec_32f * spec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);
            status = ippiResizeLinearInit_8u(srcSize, dstSize, spec);
            if ( status ) {
                std::printf("ippiResizeLinearInit_8u(Y): %d\n", status);
                return;
            }
            status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
            if ( status ) {
                std::printf("ippiResizeGetBufferSize_8u(Y): %d\n", status);
                return;
            }
            uint8_t * buf = ippsMalloc_8u(bufSize);

            status = ippiResizeLinear_8u_C1R(srcY, srcStX, dstY, dstStX, offset, dstSize, ippBorderRepl, NULL, spec, buf);
            if ( status ) {
                std::printf("ippiResizeLinear8u_C1R(Y): %d\n", status);
                return;
            }

            ippsFree(buf);
            ippsFree(spec);
        }
        // resize U and V
        {
            IppiSize srcSize = {int(srcW / 2), int(srcH / 2)};
            IppiSize dstSize = {int(dstW / 2), int(dstH / 2)};
            IppiPoint offset = {0, 0};
            int specSize = 0;
            int initBufSize = 0;
            int bufSize = 0;
            IppStatus status;
            status = ippiResizeGetSize_8u(srcSize, dstSize, ippLinear, 0, &specSize, &initBufSize);
            if ( status ) {
                std::printf("ippiResizeGetSize_8u(UV): %d\n", status);
                return;
            }
            IppiResizeSpec_32f * spec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);
            status = ippiResizeLinearInit_8u(srcSize, dstSize, spec);
            if ( status ) {
                std::printf("ippiResizeLinearInit_8u(UV): %d\n", status);
                return;
            }
            status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
            if ( status ) {
                std::printf("ippiResizeGetBufferSize_8u(Y): %d\n", status);
                return;
            }
            uint8_t * buf = ippsMalloc_8u(bufSize);

            status = ippiResizeLinear_8u_C1R(srcU, srcStX / 2, dstU, dstStX / 2, offset, dstSize, ippBorderRepl, NULL, spec, buf);
            if ( status ) {
                std::printf("ippiResizeLinear8u_C1R(U): %d\n", status);
                return;
            }
            status = ippiResizeLinear_8u_C1R(srcV, srcStX / 2, dstV, dstStX / 2, offset, dstSize, ippBorderRepl, NULL, spec, buf);
            if ( status ) {
                std::printf("ippiResizeLinear8u_C1R(V): %d\n", status);
                return;
            }

            ippsFree(buf);
            ippsFree(spec);
        }
    }

    void ResizeIPPCubic(
        int degree,
        intptr_t srcStX,
        intptr_t srcW,
        intptr_t srcStY,
        intptr_t srcH,
        const uint8_t * srcY,
        const uint8_t * srcU,
        const uint8_t * srcV,
        intptr_t dstStX,
        intptr_t dstW,
        intptr_t dstStY,
        intptr_t dstH,
        uint8_t * dstY,
        uint8_t * dstU,
        uint8_t * dstV)
    {
        (void)degree;
        (void)srcStY;
        (void)dstStY;

        // resize Y
        {
            IppiSize srcSize = {int(srcW), int(srcH)};
            IppiSize dstSize = {int(dstW), int(dstH)};
            IppiPoint offset = {0, 0};
            int specSize = 0;
            int initBufSize = 0;
            int bufSize = 0;
            IppStatus status;
            status = ippiResizeGetSize_8u(srcSize, dstSize, ippCubic, 0, &specSize, &initBufSize);
            if ( status ) {
                std::printf("ippiResizeGetSize_8u(Y): %d\n", status);
                return;
            }
            IppiResizeSpec_32f * spec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);
            uint8_t * initBuf = ippsMalloc_8u(initBufSize);
            status = ippiResizeCubicInit_8u(srcSize, dstSize, 0.f, 0.75f, spec, initBuf);
            if ( status ) {
                std::printf("ippiResizeCubicInit_8u(Y): %d\n", status);
                return;
            }
            status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
            if ( status ) {
                std::printf("ippiResizeGetBufferSize_8u(Y): %d\n", status);
                return;
            }
            uint8_t * buf = ippsMalloc_8u(bufSize);

            status = ippiResizeCubic_8u_C1R(srcY, srcStX, dstY, dstStX, offset, dstSize, ippBorderRepl, NULL, spec, buf);
            if ( status ) {
                std::printf("ippiResizeCubic8u_C1R(Y): %d\n", status);
                return;
            }

            ippsFree(initBuf);
            ippsFree(buf);
            ippsFree(spec);
        }
        // resize U and V
        {
            IppiSize srcSize = {int(srcW / 2), int(srcH / 2)};
            IppiSize dstSize = {int(dstW / 2), int(dstH / 2)};
            IppiPoint offset = {0, 0};
            int specSize = 0;
            int initBufSize = 0;
            int bufSize = 0;
            IppStatus status;
            status = ippiResizeGetSize_8u(srcSize, dstSize, ippCubic, 0, &specSize, &initBufSize);
            if ( status ) {
                std::printf("ippiResizeGetSize_8u(UV): %d\n", status);
                return;
            }
            IppiResizeSpec_32f * spec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);
            uint8_t * initBuf = ippsMalloc_8u(initBufSize);
            status = ippiResizeCubicInit_8u(srcSize, dstSize, 0.f, 0.75f, spec, initBuf);
            if ( status ) {
                std::printf("ippiResizeCubicInit_8u(UV): %d\n", status);
                return;
            }
            status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
            if ( status ) {
                std::printf("ippiResizeGetBufferSize_8u(Y): %d\n", status);
                return;
            }
            uint8_t * buf = ippsMalloc_8u(bufSize);

            status = ippiResizeCubic_8u_C1R(srcU, srcStX / 2, dstU, dstStX / 2, offset, dstSize, ippBorderRepl, NULL, spec, buf);
            if ( status ) {
                std::printf("ippiResizeCubic8u_C1R(U): %d\n", status);
                return;
            }
            status = ippiResizeCubic_8u_C1R(srcV, srcStX / 2, dstV, dstStX / 2, offset, dstSize, ippBorderRepl, NULL, spec, buf);
            if ( status ) {
                std::printf("ippiResizeCubic8u_C1R(V): %d\n", status);
                return;
            }

            ippsFree(initBuf);
            ippsFree(buf);
            ippsFree(spec);
        }
    }


    void ResizeIPPSuper(
        int degree,
        intptr_t srcStX,
        intptr_t srcW,
        intptr_t srcStY,
        intptr_t srcH,
        const uint8_t * srcY,
        const uint8_t * srcU,
        const uint8_t * srcV,
        intptr_t dstStX,
        intptr_t dstW,
        intptr_t dstStY,
        intptr_t dstH,
        uint8_t * dstY,
        uint8_t * dstU,
        uint8_t * dstV)
    {
        (void)degree;
        (void)srcStY;
        (void)dstStY;

        // resize Y
        {
            IppiSize srcSize = {int(srcW), int(srcH)};
            IppiSize dstSize = {int(dstW), int(dstH)};
            IppiPoint offset = {0, 0};
            int specSize = 0;
            int initBufSize = 0;
            int bufSize = 0;
            IppStatus status;
            status = ippiResizeGetSize_8u(srcSize, dstSize, ippSuper, 0, &specSize, &initBufSize);
            if ( status ) {
                std::printf("ippiResizeGetSize_8u(Y): %d\n", status);
                return;
            }
            IppiResizeSpec_32f * spec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);
            status = ippiResizeSuperInit_8u(srcSize, dstSize, spec);
            if ( status ) {
                std::printf("ippiResizeSuperInit_8u(Y): %d\n", status);
                return;
            }
            status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
            if ( status ) {
                std::printf("ippiResizeGetBufferSize_8u(Y): %d\n", status);
                return;
            }
            uint8_t * buf = ippsMalloc_8u(bufSize);

            status = ippiResizeSuper_8u_C1R(srcY, srcStX, dstY, dstStX, offset, dstSize, spec, buf);
            if ( status ) {
                std::printf("ippiResizeSuper8u_C1R(Y): %d\n", status);
                return;
            }

            ippsFree(buf);
            ippsFree(spec);
        }
        // resize U and V
        {
            IppiSize srcSize = {int(srcW / 2), int(srcH / 2)};
            IppiSize dstSize = {int(dstW / 2), int(dstH / 2)};
            IppiPoint offset = {0, 0};
            int specSize = 0;
            int initBufSize = 0;
            int bufSize = 0;
            IppStatus status;
            status = ippiResizeGetSize_8u(srcSize, dstSize, ippSuper, 0, &specSize, &initBufSize);
            if ( status ) {
                std::printf("ippiResizeGetSize_8u(UV): %d\n", status);
                return;
            }
            IppiResizeSpec_32f * spec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);
            status = ippiResizeSuperInit_8u(srcSize, dstSize, spec);
            if ( status ) {
                std::printf("ippiResizeSuperInit_8u(UV): %d\n", status);
                return;
            }
            status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
            if ( status ) {
                std::printf("ippiResizeGetBufferSize_8u(Y): %d\n", status);
                return;
            }
            uint8_t * buf = ippsMalloc_8u(bufSize);

            status = ippiResizeSuper_8u_C1R(srcU, srcStX / 2, dstU, dstStX / 2, offset, dstSize, spec, buf);
            if ( status ) {
                std::printf("ippiResizeSuper8u_C1R(U): %d\n", status);
                return;
            }
            status = ippiResizeSuper_8u_C1R(srcV, srcStX / 2, dstV, dstStX / 2, offset, dstSize, spec, buf);
            if ( status ) {
                std::printf("ippiResizeSuper8u_C1R(V): %d\n", status);
                return;
            }

            ippsFree(buf);
            ippsFree(spec);
        }
    }


}


int main(int argc, char *argv[])
{
    std::map<std::string, std::string> args = getArgs(argc, argv);
    int degree = std::atoi(args["d"].c_str());
    std::string method = args["m"];
    std::string srcPath = args["i"];
    std::string dstPath = args["o"];
    long srcW = std::atol(args["iw"].c_str());
    long srcH = std::atol(args["ih"].c_str());
    long dstW = std::atol(args["ow"].c_str());
    long dstH = std::atol(args["oh"].c_str());

    if ( srcPath.empty() || dstPath.empty()
        || srcW == 0 || srcH == 0
        || dstW == 0 || dstH == 0 )
    {
        std::printf("usage: resize_yuv420 [-d degree] [-m method] -i input.yuv -iw in_width -ih in_height -o output.yuv -ow out_width -oh out_height\n");
        std::printf("degree: The degree of Lanczos (ex. A=2 means Lanczos2) (default=2)\n");
        std::printf("mthod: iqo | cv_area | cv_nearest | ipp_linear | ipp_cubic | ipp_super (default=iqo)\n");
        return EINVAL;
    }

    if ( degree == 0 ) {
        degree = 2;
    }
    if ( method.empty() ) {
        method = "iqo";
    }

    size_t srcStX = srcW + srcW % 2;
    size_t srcStY = srcH + srcH % 2;
    size_t dstStX = dstW + dstW % 2;
    size_t dstStY = dstH + dstH % 2;
    size_t srcSizeY = srcStX * srcStY;
    size_t srcSizeU = srcSizeY / 4;
    size_t srcSize = srcSizeY + srcSizeU * 2;
    size_t dstSizeY = dstStX * dstStY;
    size_t dstSizeU = dstSizeY / 4;
    size_t dstSize = dstSizeY + dstSizeU * 2;
    std::vector<uint8_t> src(srcSize);
    std::vector<uint8_t> dst(dstSize);

    std::printf("quality\n");
    std::printf("  degree: %d\n", degree);
    std::printf("  method: %s\n", method.c_str());
    std::printf("input\n");
    std::printf("    path: %s\n", srcPath.c_str());
    std::printf("    size: %lux%lu\n", srcW, srcH);
    std::printf("  stride: %lux%lu\n", srcStX, srcStY);
    std::printf("output\n");
    std::printf("    path: %s\n", dstPath.c_str());
    std::printf("    size: %lux%lu\n", dstW, dstH);
    std::printf("  stride: %lux%lu\n", dstStX, dstStY);

    // read input
    {
        std::FILE * iStream = std::fopen(srcPath.c_str(), "rb");
        if ( ! iStream ) {
            int e = errno;
            std::perror("fopen");
            std::printf("Could not open \"%s\".\n", srcPath.c_str());
            return e;
        }

        size_t numRead = std::fread(&src[0], 1, srcSize, iStream);
        if ( numRead < srcSize ) {
            int e = errno;
            std::perror("fread");
            std::printf("Could not read %lu bytes.\n", srcSize);
            return e;
        }

        fclose(iStream);
    }

    // setup pointers
    const uint8_t * srcY = &src[0];
    const uint8_t * srcU = &srcY[srcSizeY];
    const uint8_t * srcV = &srcU[srcSizeU];
    uint8_t * dstY = &dst[0];
    uint8_t * dstU = &dstY[dstSizeY];
    uint8_t * dstV = &dstU[dstSizeU];
    int64_t t = now_ns();
    clock_t c = clock();

for ( int i = 0; i < 128; ++i )
    if ( method == "iqo" ) {
        ResizeIQO(
            degree,
            srcStX, srcW,
            srcStY, srcH,
            srcY, srcU, srcV,
            dstStX, dstW,
            dstStY, dstH,
            dstY, dstU, dstV);
    } else if ( method == "cv_area" ) {
        ResizeCV(
            degree,
            srcStX, srcW,
            srcStY, srcH,
            srcY, srcU, srcV,
            dstStX, dstW,
            dstStY, dstH,
            dstY, dstU, dstV,
            cv::INTER_AREA);
    } else if ( method == "cv_nearest" ) {
        ResizeCV(
            degree,
            srcStX, srcW,
            srcStY, srcH,
            srcY, srcU, srcV,
            dstStX, dstW,
            dstStY, dstH,
            dstY, dstU, dstV,
            cv::INTER_NEAREST);
    } else if ( method == "ipp_linear" ) {
        ResizeIPPLinear(
            degree,
            srcStX, srcW,
            srcStY, srcH,
            srcY, srcU, srcV,
            dstStX, dstW,
            dstStY, dstH,
            dstY, dstU, dstV);
    } else if ( method == "ipp_cubic" ) {
        ResizeIPPCubic(
            degree,
            srcStX, srcW,
            srcStY, srcH,
            srcY, srcU, srcV,
            dstStX, dstW,
            dstStY, dstH,
            dstY, dstU, dstV);
    } else if ( method == "ipp_super" ) {
        ResizeIPPSuper(
            degree,
            srcStX, srcW,
            srcStY, srcH,
            srcY, srcU, srcV,
            dstStX, dstW,
            dstStY, dstH,
            dstY, dstU, dstV);
    } else {
        std::printf("Unknown method %s\n", method.c_str());
        return EINVAL;
    }

    t = now_ns() - t;
    c = std::clock() - c;
    std::printf("CPU Time: %9.6fs\n", double(c) / CLOCKS_PER_SEC);
    std::printf("    Time: %9.6fs\n", double(t) / 1000000000);

    // write output
    {
        std::FILE * oStream = std::fopen(dstPath.c_str(), "wb");
        if ( ! oStream ) {
            int e = errno;
            std::perror("fopen");
            std::printf("Could not open \"%s\".\n", dstPath.c_str());
            return e;
        }

        size_t numWrite = std::fwrite(&dst[0], 1, dstSize, oStream);
        if ( numWrite < dstSize ) {
            int e = errno;
            std::perror("fwrite");
            std::printf("Could not write %lu bytes.\n", dstSize);
            return e;
        }

        fclose(oStream);
    }

    return 0;
}
