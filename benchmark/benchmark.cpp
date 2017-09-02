#include <stdint.h>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <memory>

#include <libiqo/iqo.hpp>

#include "config.h"

#if defined(_OPENMP)
    #include <omp.h>
#endif

#if defined(HAVE_OPENCV_HPP)
#   include <opencv2/opencv.hpp>
#endif

#if defined(HAVE_IPP_H)
#   include <ipp.h>
#endif

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

    void fillRandom(uint8_t * __restrict p, size_t size)
    {
        std::mt19937 rand_src(0);
        std::uniform_int_distribution<int> rand_dist(0, 255);

        for ( size_t i = 0; i < size; i++ ) {
            p[i] = uint8_t(rand_dist(rand_src));
        }
    }

    bool haveOpenCV()
    {
#if defined(HAVE_OPENCV_HPP)
        return true;
#else
        return false;
#endif
    }

    bool haveIPP()
    {
#if defined(HAVE_IPP_H)
        return true;
#else
        return false;
#endif
    }

    int IQOGetNumberOfThreads()
    {
#if defined(_OPENMP)
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    int CVGetNumThreads()
    {
#if defined(HAVE_OPENCV_HPP)
        return cv::getNumThreads();
#else
        return 0;
#endif
    }

    int IPPGetNumThreads()
    {
#if defined(HAVE_IPP_H)
        int n = 0;
        ippGetNumThreads(&n);
        return n;
#else
        return 0;
#endif
    }

    class IResizer
    {
    public:
        virtual int getNumThreads() const = 0;
        virtual int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) = 0;
    };

    //! IQO Area
    class IQOAreaResizer : public IResizer
    {
    public:
        int getNumThreads() const
        {
            return IQOGetNumberOfThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            // resize Y
            {
                iqo::AreaResizer r(srcW, srcH, dstW, dstH);

                r.resize(srcStY, srcY, dstStY, dstY);
            }
            // resize U and V
            {
                iqo::AreaResizer r(srcW / 2, srcH / 2, dstW / 2, dstH / 2);

                r.resize(srcStUV, srcU, dstStUV, dstU);
                r.resize(srcStUV, srcV, dstStUV, dstV);
            }

            return 0;
        }
    };

    //! IQO Linear
    class IQOLinearResizer : public IResizer
    {
    public:
        int getNumThreads() const
        {
            return IQOGetNumberOfThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            // resize Y
            {
                iqo::LinearResizer r(srcW, srcH, dstW, dstH);

                r.resize(srcStY, srcY, dstStY, dstY);
            }
            // resize U and V
            {
                iqo::LinearResizer r(srcW / 2, srcH / 2, dstW / 2, dstH / 2);

                r.resize(srcStUV, srcU, dstStUV, dstU);
                r.resize(srcStUV, srcV, dstStUV, dstV);
            }

            return 0;
        }
    };

    //! IQO Lanczos
    class IQOLanczosResizer : public IResizer
    {
    public:
        IQOLanczosResizer(int degree)
        {
            m_Degree = degree;
        }

        int getNumThreads() const
        {
            return IQOGetNumberOfThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            // resize Y
            {
                iqo::LanczosResizer r(m_Degree, srcW, srcH, dstW, dstH, 1);

                r.resize(srcStY, srcY, dstStY, dstY);
            }
            // resize U and V
            {
                iqo::LanczosResizer r(m_Degree, srcW / 2, srcH / 2, dstW / 2, dstH / 2, 2);

                r.resize(srcStUV, srcU, dstStUV, dstU);
                r.resize(srcStUV, srcV, dstStUV, dstV);
            }

            return 0;
        }

    private:
        int m_Degree;
    };

    int ResizeCV420P(
        size_t srcW, size_t srcH,
        size_t srcStY, const uint8_t * srcY,
        size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
        size_t dstW, size_t dstH,
        size_t dstStY, uint8_t * dstY,
        size_t dstStUV, uint8_t * dstU, uint8_t * dstV,
        int algorithm
    ) {
        (void)srcStY;
        (void)srcStUV;
        (void)dstStY;
        (void)dstStUV;

        int status = 0;

#if defined(HAVE_OPENCV_HPP)
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
#else
        std::printf("opencv is not implemented.\n");
        status = 1;
#endif
        return status;
    }

    //! OpenCV Area
    class CVAreaResizer : public IResizer
    {
    public:
        int getNumThreads() const
        {
            return CVGetNumThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            int algorithm = 0;
#if defined(HAVE_OPENCV_HPP)
            algorithm = cv::INTER_AREA;
#endif
            return ResizeCV420P(
                srcW, srcH, srcStY, srcY, srcStUV, srcU, srcV,
                dstW, dstH, dstStY, dstY, dstStUV, dstU, dstV,
                algorithm
            );
        }
    };

    //! OpenCV Linear
    class CVLinearResizer : public IResizer
    {
    public:
        int getNumThreads() const
        {
            return CVGetNumThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            int algorithm = 0;
#if defined(HAVE_OPENCV_HPP)
            algorithm = cv::INTER_LINEAR;
#endif
            return ResizeCV420P(
                srcW, srcH, srcStY, srcY, srcStUV, srcU, srcV,
                dstW, dstH, dstStY, dstY, dstStUV, dstU, dstV,
                algorithm
            );
        }
    };

    //! OpenCV Lanczos4
    class CVLanczos4Resizer : public IResizer
    {
    public:
        int getNumThreads() const
        {
            return CVGetNumThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            int algorithm = 0;
#if defined(HAVE_OPENCV_HPP)
            algorithm = cv::INTER_LANCZOS4;
#endif
            return ResizeCV420P(
                srcW, srcH, srcStY, srcY, srcStUV, srcU, srcV,
                dstW, dstH, dstStY, dstY, dstStUV, dstU, dstV,
                algorithm
            );
        }
    };

    //! IPP Super Sampling 
    class IPPSuperResizer : public IResizer
    {
    public:
        int getNumThreads() const
        {
            return IPPGetNumThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            int status = 0;

#if defined(HAVE_IPP_H)
            // resize Y
            {
                IppiSize srcSize = { Ipp32s(srcW), Ipp32s(srcH) };
                IppiSize dstSize = { Ipp32s(dstW), Ipp32s(dstH) };
                IppiPoint offset = { 0, 0 };
                Ipp32s specSize = 0;
                Ipp32s initBufSize = 0; // not used
                Ipp32s bufSize = 0;
                IppiResizeSpec_32f * spec = nullptr;
                Ipp8u * buffer = nullptr;

                status = ippiResizeGetSize_8u(srcSize, dstSize, ippSuper, 0, &specSize, &initBufSize);
                if ( status ) {
                    std::printf("Y: error ippiResizeGetSize_8u: %d\n", status);
                    goto failY;
                }
                spec = reinterpret_cast<IppiResizeSpec_32f*>(ippsMalloc_8u(specSize));

                status = ippiResizeSuperInit_8u(srcSize, dstSize, spec);
                if ( status ) {
                    std::printf("Y: error ippiResizeSuperInit_8u: %d\n", status);
                    goto failY;
                }

                status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
                if ( status ) {
                    std::printf("Y: error ippiResizeGetBufferSize_8u: %d\n", status);
                    goto failY;
                }
                buffer = ippsMalloc_8u(bufSize);

                status = ippiResizeSuper_8u_C1R(
                    srcY, Ipp32s(srcStY),
                    dstY, Ipp32s(dstStY),
                    offset, dstSize,
                    spec, buffer
                );
                if ( status ) {
                    std::printf("Y: error ippiResizeSuper_8u_C1R: %d\n", status);
                    goto failY;
                }

failY:
                ippsFree(buffer);
                ippsFree(spec);
            }
            if ( status ) {
                return status;
            }

            // resize UV
            {
                IppiSize srcSize = { Ipp32s(srcW / 2), Ipp32s(srcH / 2) };
                IppiSize dstSize = { Ipp32s(dstW / 2), Ipp32s(dstH / 2) };
                IppiPoint offset = { 0, 0 };
                Ipp32s specSize = 0;
                Ipp32s initBufSize = 0; // not used
                Ipp32s bufSize = 0;
                IppiResizeSpec_32f * spec = nullptr;
                Ipp8u * buffer = nullptr;

                status = ippiResizeGetSize_8u(srcSize, dstSize, ippSuper, 0, &specSize, &initBufSize);
                if ( status ) {
                    std::printf("UV: error ippiResizeGetSize_8u: %d\n", status);
                    goto failUV;
                }
                spec = reinterpret_cast<IppiResizeSpec_32f*>(ippsMalloc_8u(specSize));

                status = ippiResizeSuperInit_8u(srcSize, dstSize, spec);
                if ( status ) {
                    std::printf("UV: error ippiResizeSuperInit_8u: %d\n", status);
                    goto failUV;
                }

                status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
                if ( status ) {
                    std::printf("UV: error ippiResizeGetBufferSize_8u: %d\n", status);
                    goto failUV;
                }
                buffer = ippsMalloc_8u(bufSize);

                status = ippiResizeSuper_8u_C1R(
                    srcU, Ipp32s(srcStUV),
                    dstU, Ipp32s(dstStUV),
                    offset, dstSize,
                    spec, buffer
                );
                if ( status ) {
                    std::printf("U: error ippiResizeSuper_8u_C1R: %d\n", status);
                    goto failUV;
                }

                status = ippiResizeSuper_8u_C1R(
                    srcV, Ipp32s(srcStUV),
                    dstV, Ipp32s(dstStUV),
                    offset, dstSize,
                    spec, buffer
                );
                if ( status ) {
                    std::printf("V: error ippiResizeSuper_8u_C1R: %d\n", status);
                    goto failUV;
                }

failUV:
                ippsFree(buffer);
                ippsFree(spec);
            }
#else
            std::printf("ipp-super is not implemented.\n");
            status = 1;
#endif

            return status;
        }
    };

    //! IPP Linear
    class IPPLinearResizer : public IResizer
    {
    public:
        int getNumThreads() const
        {
            return IPPGetNumThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            int status = 0;

#if defined(HAVE_IPP_H)
            // resize Y
            {
                IppiSize srcSize = { Ipp32s(srcW), Ipp32s(srcH) };
                IppiSize dstSize = { Ipp32s(dstW), Ipp32s(dstH) };
                IppiPoint offset = { 0, 0 };
                Ipp32s specSize = 0;
                Ipp32s initBufSize = 0;
                Ipp32s bufSize = 0;
                IppiResizeSpec_32f * spec = nullptr;
                Ipp8u * initBuffer = nullptr;
                Ipp8u * buffer = nullptr;
                Ipp32u isAA = (srcSize.width > dstSize.width || srcSize.height > dstSize.height);

                status = ippiResizeGetSize_8u(srcSize, dstSize, ippLinear, isAA, &specSize, &initBufSize);
                if ( status ) {
                    std::printf("Y: error ippiResizeGetSize_8u: %d\n", status);
                    goto failY;
                }
                spec = reinterpret_cast<IppiResizeSpec_32f*>(ippsMalloc_8u(specSize));

                if ( isAA ) {
                    initBuffer = ippsMalloc_8u(initBufSize);
                    status = ippiResizeAntialiasingLinearInit(srcSize, dstSize, spec, initBuffer);
                } else {
                    status = ippiResizeLinearInit_8u(srcSize, dstSize, spec);
                }
                if ( status ) {
                    std::printf("Y: error ippiResizeLinearInit_8u: %d\n", status);
                    goto failY;
                }

                status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
                if ( status ) {
                    std::printf("Y: error ippiResizeGetBufferSize_8u: %d\n", status);
                    goto failY;
                }
                buffer = ippsMalloc_8u(bufSize);

                if ( isAA ) {
                    status = ippiResizeAntialiasing_8u_C1R(
                        srcY, Ipp32s(srcStY),
                        dstY, Ipp32s(dstStY),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                } else {
                    status = ippiResizeLinear_8u_C1R(
                        srcY, Ipp32s(srcStY),
                        dstY, Ipp32s(dstStY),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                }
                if ( status ) {
                    std::printf("Y: error ippiResizeLinear_8u_C1R: %d\n", status);
                    goto failY;
                }

failY:
                ippsFree(buffer);
                ippsFree(initBuffer);
                ippsFree(spec);
            }
            if ( status ) {
                return status;
            }

            // resize UV
            {
                IppiSize srcSize = { Ipp32s(srcW / 2), Ipp32s(srcH / 2) };
                IppiSize dstSize = { Ipp32s(dstW / 2), Ipp32s(dstH / 2) };
                IppiPoint offset = { 0, 0 };
                Ipp32s specSize = 0;
                Ipp32s initBufSize = 0;
                Ipp32s bufSize = 0;
                IppiResizeSpec_32f * spec = nullptr;
                Ipp8u * initBuffer = nullptr;
                Ipp8u * buffer = nullptr;
                Ipp32u isAA = (srcSize.width > dstSize.width || srcSize.height > dstSize.height);

                status = ippiResizeGetSize_8u(srcSize, dstSize, ippLinear, isAA, &specSize, &initBufSize);
                if ( status ) {
                    std::printf("UV: error ippiResizeGetSize_8u: %d\n", status);
                    goto failUV;
                }
                spec = reinterpret_cast<IppiResizeSpec_32f*>(ippsMalloc_8u(specSize));
                initBuffer = ippsMalloc_8u(initBufSize);

                if ( isAA ) {
                    status = ippiResizeAntialiasingLinearInit(srcSize, dstSize, spec, initBuffer);
                } else {
                    status = ippiResizeLinearInit_8u(srcSize, dstSize, spec);
                }
                if ( status ) {
                    std::printf("UV: error ippiResizeLinearInit_8u: %d\n", status);
                    goto failUV;
                }

                status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
                if ( status ) {
                    std::printf("UV: error ippiResizeGetBufferSize_8u: %d\n", status);
                    goto failUV;
                }
                buffer = ippsMalloc_8u(bufSize);

                if ( isAA ) {
                    status = ippiResizeAntialiasing_8u_C1R(
                        srcU, Ipp32s(srcStUV),
                        dstU, Ipp32s(dstStUV),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                } else {
                    status = ippiResizeLinear_8u_C1R(
                        srcU, Ipp32s(srcStUV),
                        dstU, Ipp32s(dstStUV),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                }
                if ( status ) {
                    std::printf("U: error ippiResizeLinear_8u_C1R: %d\n", status);
                    goto failUV;
                }

                if ( isAA ) {
                    status = ippiResizeAntialiasing_8u_C1R(
                        srcV, Ipp32s(srcStUV),
                        dstV, Ipp32s(dstStUV),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                } else {
                    status = ippiResizeLinear_8u_C1R(
                        srcV, Ipp32s(srcStUV),
                        dstV, Ipp32s(dstStUV),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                }
                if ( status ) {
                    std::printf("V: error ippiResizeLinear_8u_C1R: %d\n", status);
                    goto failUV;
                }

failUV:
                ippsFree(buffer);
                ippsFree(initBuffer);
                ippsFree(spec);
            }
#else
            std::printf("ipp-super is not implemented.\n");
            status = 1;
#endif

            return status;
        }
    };

    //! IPP Super Sampling 
    class IPPLanczosResizer : public IResizer
    {
    public:
        IPPLanczosResizer(int degree)
        {
            m_Degree = degree;
        }

        int getNumThreads() const
        {
            return IPPGetNumThreads();
        }

        int resize(
            size_t srcW, size_t srcH,
            size_t srcStY, const uint8_t * srcY,
            size_t srcStUV, const uint8_t * srcU, const uint8_t * srcV,
            size_t dstW, size_t dstH,
            size_t dstStY, uint8_t * dstY,
            size_t dstStUV, uint8_t * dstU, uint8_t * dstV
        ) {
            int status = 0;

#if defined(HAVE_IPP_H)
            // resize Y
            {
                IppiSize srcSize = { Ipp32s(srcW), Ipp32s(srcH) };
                IppiSize dstSize = { Ipp32s(dstW), Ipp32s(dstH) };
                IppiPoint offset = { 0, 0 };
                Ipp32s specSize = 0;
                Ipp32s initBufSize = 0;
                Ipp32s bufSize = 0;
                IppiResizeSpec_32f * spec = nullptr;
                Ipp8u * initBuffer = nullptr;
                Ipp8u * buffer = nullptr;
                Ipp32u isAA = (srcSize.width > dstSize.width || srcSize.height > dstSize.height);

                status = ippiResizeGetSize_8u(srcSize, dstSize, ippLanczos, isAA, &specSize, &initBufSize);
                if ( status ) {
                    std::printf("Y: error ippiResizeGetSize_8u: %d\n", status);
                    goto failY;
                }
                spec = reinterpret_cast<IppiResizeSpec_32f*>(ippsMalloc_8u(specSize));
                initBuffer = ippsMalloc_8u(initBufSize);

                if ( isAA ) {
                    status = ippiResizeAntialiasingLanczosInit(srcSize, dstSize, m_Degree, spec, initBuffer);
                } else {
                    status = ippiResizeLanczosInit_8u(srcSize, dstSize, m_Degree, spec, initBuffer);
                }
                if ( status ) {
                    std::printf("Y: error ippiResizeLanczosInit_8u: %d\n", status);
                    goto failY;
                }

                status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
                if ( status ) {
                    std::printf("Y: error ippiResizeGetBufferSize_8u: %d\n", status);
                    goto failY;
                }
                buffer = ippsMalloc_8u(bufSize);

                if ( isAA ) {
                    status = ippiResizeAntialiasing_8u_C1R(
                        srcY, Ipp32s(srcStY),
                        dstY, Ipp32s(dstStY),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                } else {
                    status = ippiResizeLanczos_8u_C1R(
                        srcY, Ipp32s(srcStY),
                        dstY, Ipp32s(dstStY),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                }
                if ( status ) {
                    std::printf("Y: error ippiResizeLanczos_8u_C1R: %d\n", status);
                    goto failY;
                }

failY:
                ippsFree(buffer);
                ippsFree(initBuffer);
                ippsFree(spec);
            }
            if ( status ) {
                return status;
            }

            // resize UV
            {
                IppiSize srcSize = { Ipp32s(srcW / 2), Ipp32s(srcH / 2) };
                IppiSize dstSize = { Ipp32s(dstW / 2), Ipp32s(dstH / 2) };
                IppiPoint offset = { 0, 0 };
                Ipp32s specSize = 0;
                Ipp32s initBufSize = 0;
                Ipp32s bufSize = 0;
                IppiResizeSpec_32f * spec = nullptr;
                Ipp8u * initBuffer = nullptr;
                Ipp8u * buffer = nullptr;
                Ipp32u isAA = (srcSize.width > dstSize.width || srcSize.height > dstSize.height);

                status = ippiResizeGetSize_8u(srcSize, dstSize, ippLanczos, isAA, &specSize, &initBufSize);
                if ( status ) {
                    std::printf("UV: error ippiResizeGetSize_8u: %d\n", status);
                    goto failUV;
                }
                spec = reinterpret_cast<IppiResizeSpec_32f*>(ippsMalloc_8u(specSize));
                initBuffer = ippsMalloc_8u(initBufSize);

                if ( isAA ) {
                    status = ippiResizeAntialiasingLanczosInit(srcSize, dstSize, m_Degree, spec, initBuffer);
                } else {
                    status = ippiResizeLanczosInit_8u(srcSize, dstSize, m_Degree, spec, initBuffer);
                }
                if ( status ) {
                    std::printf("UV: error ippiResizeLanczosInit_8u: %d\n", status);
                    goto failUV;
                }

                status = ippiResizeGetBufferSize_8u(spec, dstSize, 1, &bufSize);
                if ( status ) {
                    std::printf("UV: error ippiResizeGetBufferSize_8u: %d\n", status);
                    goto failUV;
                }
                buffer = ippsMalloc_8u(bufSize);

                if ( isAA ) {
                    status = ippiResizeAntialiasing_8u_C1R(
                        srcU, Ipp32s(srcStUV),
                        dstU, Ipp32s(dstStUV),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                } else {
                    status = ippiResizeLanczos_8u_C1R(
                        srcU, Ipp32s(srcStUV),
                        dstU, Ipp32s(dstStUV),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                }
                if ( status ) {
                    std::printf("U: error ippiResizeLanczos_8u_C1R: %d\n", status);
                    goto failUV;
                }

                if ( isAA ) {
                    status = ippiResizeAntialiasing_8u_C1R(
                        srcV, Ipp32s(srcStUV),
                        dstV, Ipp32s(dstStUV),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                } else {
                    status = ippiResizeLanczos_8u_C1R(
                        srcV, Ipp32s(srcStUV),
                        dstV, Ipp32s(dstStUV),
                        offset, dstSize,
                        ippBorderRepl, nullptr,
                        spec, buffer
                    );
                }
                if ( status ) {
                    std::printf("V: error ippiResizeLanczos_8u_C1R: %d\n", status);
                    goto failUV;
                }

failUV:
                ippsFree(buffer);
                ippsFree(initBuffer);
                ippsFree(spec);
            }
#else
            std::printf("ipp-super is not implemented.\n");
            status = 1;
#endif

            return status;
        }

    private:
        int m_Degree;
    };

}


int main(int argc, char *argv[])
{
#if defined(HAVE_IPP_H)
    ippInit();
#endif

    std::map<std::string, std::string> args = getArgs(argc, argv);
    std::string method = args["m"];
    long srcW = std::atol(args["iw"].c_str());
    long srcH = std::atol(args["ih"].c_str());
    long dstW = std::atol(args["ow"].c_str());
    long dstH = std::atol(args["oh"].c_str());
    int degree = 2;
    int numCycles = 1000;

    if (   srcW == 0 || srcH == 0
        || dstW == 0 || dstH == 0 )
    {
        std::printf("usage: benchmark -m method -iw in_width -ih in_height -ow out_width -oh out_height\n");
        std::printf("method: area | linear | lanczos[1-9]");
        if ( haveOpenCV() ) {
            std::printf(" | cv-area | cv-linear | cv-lanczos4");
        }
        if ( haveIPP() ) {
            std::printf(" | ipp-super | ipp-linear | ipp-lanczos[2-3]");
        }
        std::printf("\n");
        return EINVAL;
    }

    if ( method.size() == std::strlen("lanczos1") && method.find("lanczos") == 0 ) {
        degree = method[7] - '0';
        if ( degree < 1 || 9 < degree ) {
            std::printf("invalid method: %s\n", method.c_str());
            return EINVAL;
        }
        method = "lanczos";
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
    std::unique_ptr<IResizer> resizer;

    std::printf("method: %s\n", method.c_str());
    if ( method == "lanczos" ) {
        std::printf("quality\n");
        std::printf("  degree: %d\n", degree);
    }

    if ( method == "area" ) {
        if ( srcW < dstW || srcH < dstH ) {
            std::printf("warning: area supports only down-sampling.\n");
        }
        resizer.reset(new IQOAreaResizer);
    }
    if ( method == "linear" ) {
        if ( srcW > dstW || srcH > dstH ) {
            std::printf("warning: linear supports only up-sampling.\n");
        }
        resizer.reset(new IQOLinearResizer);
    }
    if ( method == "lanczos" ) {
        resizer.reset(new IQOLanczosResizer(degree));
    }
    if ( method == "cv-area" ) {
        if ( srcW < dstW || srcH < dstH ) {
            std::printf("warning: cv-area supports only down-sampling.\n");
        }
        resizer.reset(new CVAreaResizer);
    }
    if ( method == "cv-linear" ) {
        if ( srcW > dstW || srcH > dstH ) {
            std::printf("warning: cv-linear supports only up-sampling.\n");
        }
        resizer.reset(new CVLinearResizer);
    }
    if ( method == "cv-lanczos4" ) {
        if ( srcW > dstW || srcH > dstH ) {
            std::printf("warning: cv-lanczos4 supports only up-sampling.\n");
        }
        resizer.reset(new CVLanczos4Resizer);
    }
    if ( method == "ipp-super" ) {
        if ( srcW < dstW || srcH < dstH ) {
            std::printf("warning: ipp-super supports only down-sampling.\n");
        }
        resizer.reset(new IPPSuperResizer);
    }
    if ( method == "ipp-linear" ) {
        resizer.reset(new IPPLinearResizer);
    }
    if ( method == "ipp-lanczos2" ) {
        resizer.reset(new IPPLanczosResizer(2));
    }
    if ( method == "ipp-lanczos3" ) {
        resizer.reset(new IPPLanczosResizer(3));
    }
    if ( ! resizer ) {
        std::printf("invalid method: %s\n", method.c_str());
        return EINVAL;
    }

    std::printf("cpu\n");
    std::printf(" threads: %d\n", resizer->getNumThreads());
    std::printf("input\n");
    std::printf("    size: %lux%lu\n", srcW, srcH);
    std::printf("  stride: %lux%lu\n", srcStX, srcStY);
    std::printf("output\n");
    std::printf("    size: %lux%lu\n", dstW, dstH);
    std::printf("  stride: %lux%lu\n", dstStX, dstStY);
    std::printf("benchmark\n");
    std::printf("  cycles: %d\n", numCycles);

    // setup pointers
    uint8_t * srcY = &src[0];
    uint8_t * srcU = &srcY[srcSizeY];
    uint8_t * srcV = &srcU[srcSizeU];
    uint8_t * dstY = &dst[0];
    uint8_t * dstU = &dstY[dstSizeY];
    uint8_t * dstV = &dstU[dstSizeU];

    fillRandom(srcY, srcSizeY);
    fillRandom(srcU, srcSizeU);
    fillRandom(srcV, srcSizeU);

    auto startTime = std::chrono::high_resolution_clock::now();

    for ( int i = 0; i < numCycles; i++ ) {
        int status = resizer->resize(
            srcW, srcH, srcStX, srcY, srcStX / 2, srcU, srcV,
            dstW, dstH, dstStX, dstY, dstStX / 2, dstU, dstV
        );
        if ( status ) {
            return EINVAL;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    std::printf("  elapsed time: %6.3f secs\n", elapsedTime.count());

    return 0;
}
