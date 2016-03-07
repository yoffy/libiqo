#include <stdint.h>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <libiqo/IQOLanczosResizer.hpp>

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

}


int main(int argc, char *argv[])
{
    std::map<std::string, std::string> args = getArgs(argc, argv);
    int degree = std::atoi(args["d"].c_str());
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
        std::printf("usage: resize_yuv420 [-d degree] -i input.yuv -iw in_width -ih in_height -o output.yuv -ow out_width -oh out_height\n");
        std::printf("degree The degree of Lanczos (ex. A=2 means Lanczos2) (default=2)\n");
        return EINVAL;
    }

    if ( degree == 0 ) {
        degree = 2;
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

    // resize Y
    {
        iqo::LanczosResizer r(degree, srcW, srcH, dstW, dstH);

        r.resize(srcStX, srcY, dstStX, dstY);
    }
    // resize U and V
    {
        iqo::LanczosResizer r(degree, srcStX / 2, srcStY / 2, dstStX / 2, dstStY / 2);

        r.resize(srcStX / 2, srcU, dstStX / 2, dstU);
        r.resize(srcStX / 2, srcV, dstStX / 2, dstV);
    }

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