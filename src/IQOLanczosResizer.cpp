#include <stdint.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <immintrin.h>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "libiqo/IQOLanczosResizer.hpp"

namespace {

    int getNumberOfProcs()
    {
#if defined(_OPENMP)
        return omp_get_num_procs();
#else
        return 1;
#endif
    }

    int getThreadNumber()
    {
#if defined(_OPENMP)
        return omp_get_thread_num();
#else
        return 0;
#endif
    }

    intptr_t alignFloor(intptr_t v, intptr_t alignment)
    {
        return v / alignment * alignment;
    }

    intptr_t alignCeil(intptr_t v, intptr_t alignment)
    {
        return (v + (alignment - 1)) / alignment * alignment;
    }

    template<typename T>
    T sinc(T x)
    {
        T fPi  = 3.14159265358979;
        T fPiX = fPi * x;
        return std::sin(fPiX) / fPiX;
    }

    template<typename T>
    T lanczos(int degree, T x)
    {
        T absX = std::fabs(x);
        if ( std::fmod(absX, T(1)) < T(1e-5) ) {
            return absX < T(1e-5) ? 1 : 0;
        }
        if ( degree <= absX ) {
            return 0;
        }
        return sinc(x) * sinc(x / degree);
    }

    //! @brief Set Lanczos table
    //! @param degree     Degree of Lanczos (ex. A=2 means Lanczos2)
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param dstOffset  The coordinate of the destination image
    //! @param pxScale  Scale of a pixel (ex. 2 when U plane of YUV420 image)
    //! @param n          Size of table
    //! @param fTable     The table
    //! @return Sum of the table
    //!
    //! Calculate Lanczos coefficients from `-degree` to `+degree`.
    //!
    //! n should be `2*degree` when up sampling.
    template<typename T>
    T setLanczosTable(
        int degree,
        intptr_t srcLen,
        intptr_t dstLen,
        intptr_t dstOffset,
        intptr_t pxScale,
        int n,
        T * fTable)
    {
        // down-sampling (ex. lanczos3)
        //
        // case decimal number of coefs:
        //        scale = 5:4
        // num of coefs = 3*5/4 = 3.75
        //
        //       start = -3
        //  num pixels =  4
        // -3           0
        //  o   o   o   o   o   o   o   o   o   o   o
        //  o    o    o    o    o    o    o    o    o
        //  +--------------+
        //
        //                       start = -3
        //                  num pixels =  4
        //                 -3           0
        //  o   o   o   o   o   o   o   o   o   o   o
        //  o    o    o    o    o    o    o    o    o
        //                 +--------------+
        //
        // case integer number of coefs:
        //        scale = 4:3
        // num of coefs = 3*4/3 = 4
        //
        //       start = -4 -> -3
        //  num pixels =  5 ->  4 (correct into num of coefs)
        // -4           0
        //  o  o  o  o  o  o  o  o  o
        //  o   o   o   o   o   o   o
        //  +-----------+
        //
        //                start = -3
        //           num pixels =  4
        //          -3        0
        //  o  o  o  o  o  o  o  o  o
        //  o   o   o   o   o   o   o
        //          +-----------+
        //
        // case integer number of coefs:
        //        scale = 8:3
        // num of coefs = 3*8/3 = 8
        //
        //       start = -8 -> -7
        //  num pixels =  9 ->  8 (correct into num of coefs)
        // -8             0
        //  o o o o o o o o o o o o o o o o
        //  o      o      o      o      o
        //  +-------------+

        //   o: center of a pixel (on a coordinate)
        //   |: boundary of pixels
        //
        // Theoretical space (aligned to center of a pixel):
        // start:     -degree + std::fmod((1 - srcOffset) * scale, 1.0)
        //             v
        //   src:    | o | o | o | o | o | o | o | o | o | o | o | o | o |
        //   dst:  |   o   |   o   |   o   |   o   |   o   |   o   |   o   |
        //             3       2       1       0       1       2       3
        //             ^
        //             degree
        //
        // Display space (aligned to boundary of pixels):
        // start:     (-degree - 0.5 + 0.5*scale) + std::fmod((1 - srcOffset) * scale, 1.0)
        //             v
        //   src:   |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |  o  |
        //   dst:   |  : o     |    o     |    o     |     o     |     o    |     o    |     o    |
        //         3.5 : 3          2          1           0           1          2          3
        //          ^  :
        // degree-0.5  0.5*scale

        // X is offset of Lanczos from center.
        // It will be source coordinate when up-sampling,
        // or, destination coordinate when down-sampling.
        double beginX = 0;
        if ( srcLen > dstLen ) {
            // down-sampling

            //----- easy solution -----
            //double scale = dstLen / double(srcLen);
            //double srcOffset = std::fmod(dstOffset / scale, 1.0);
            //double beginX = -degree + (-0.5 + 0.5*scale)*pxScale + std::fmod((1 - srcOffset) * scale * pxScale, 1.0);

            //----- more accurate -----
            // srcOffset = std::fmod(dstOffset / scale, 1.0)
            //           = (dstOffset * srcLen % dstLen) / dstLen;

            // -degree + (-0.5 + 0.5*scale)*pxScale
            // = -degree + (-0.5 + 0.5*dstLen/srcLen) * pxScale
            // = -degree - 0.5*pxScale + 0.5*dstLen*pxScale/srcLen

            // std::fmod((1 - srcOffset) * scale * pxScale, 1.0)
            // = std::fmod((1 - (dstOffset * srcLen % dstLen)/dstLen) * (dstLen/srcLen) * pxScale, 1.0)
            // = std::fmod((dstLen/srcLen - (dstOffset * srcLen % dstLen)/dstLen*(dstLen/srcLen)) * pxScale, 1.0)
            // = std::fmod((dstLen/srcLen - (dstOffset * srcLen % dstLen)/srcLen) * pxScale, 1.0)
            // = std::fmod(((dstLen - (dstOffset * srcLen % dstLen))/srcLen) * pxScale, 1.0)
            // = std::fmod( (dstLen - (dstOffset * srcLen % dstLen))         * pxScale, srcLen) / srcLen
            // = ((dstLen - dstOffset*srcLen%dstLen) * pxScale % srcLen) / srcLen
            int degFactor = std::max<int>(1, pxScale / degree);
            beginX =
                -degree*degFactor - 0.5*pxScale + 0.5*dstLen*pxScale/srcLen
                + ((dstLen - dstOffset * srcLen % dstLen) * pxScale % srcLen) / double(srcLen);
        } else {
            // up-sampling
            double srcOffset = std::fmod(dstOffset * srcLen / double(dstLen), 1.0);
            beginX = -degree + 1.0 - srcOffset;
            srcLen = dstLen; // scale = 1.0
            pxScale = 1;
        }

        T fSum = 0;

        for ( intptr_t i = 0; i < n; ++i ) {
            //     x = beginX + i * scale * pxScale
            double x = beginX + (i * dstLen * pxScale) / double(srcLen);
            T v = T(lanczos(degree, x));
            fTable[i] = v;
            fSum     += v;
        }

        return fSum;
    }

    template<typename T>
    T round(T x)
    {
        return std::floor(x + T(0.5));
    }

    template<typename T>
    T clamp(T lo, T hi, T v)
    {
        return std::max(lo, std::min(hi, v));
    }

    intptr_t gcd(intptr_t a, intptr_t b)
    {
        intptr_t r = a % b;

        while ( r ) {
            a = b;
            b = r;
            r = a % b;
        }

        return b;
    }

    intptr_t lcm(intptr_t a, intptr_t b)
    {
        return a * b / gcd(a, b);
    }

    //! (uint8_t)min(255, max(0, v))
    __m128i packus_epi16(__m256i v)
    {
        __m128i s16x8L = _mm256_castsi256_si128(v);
        __m128i s16x8H = _mm256_extractf128_si256(v, 1);
        return _mm_packus_epi16(s16x8L, s16x8H);
    }

    __m256i packus_epi32(__m256i lo, __m256i hi)
    {
        __m256i u16x16Perm = _mm256_packus_epi32(lo, hi);
        return _mm256_permute4x64_epi64(u16x16Perm, _MM_SHUFFLE(3, 1, 2, 0));
    }

    //! (uint8_t)min(255, max(0, round(v)))
    __m128i cvtrps_epu8(__m256i lo, __m256i hi)
    {
        __m256  f32x8L = _mm256_round_ps(lo, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256  f32x8H = _mm256_round_ps(hi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i s32x8L = _mm256_cvttps_epi32(f32x8L);
        __m256i s32x8H = _mm256_cvttps_epi32(f32x8H);
        __m256i u16x16 = packus_epi32(s32x8L, s32x8H);
        return packus_epi16(u16x16);
    }

}

namespace iqo {

    class LanczosResizer::Impl
    {
    public:
        //! Constructor
        Impl(unsigned int degree, size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t pxScale);

        //! Run image resizing
        void resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * dst);

    private:
        void resizeYborder(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, float * dst,
            intptr_t srcOY,
            const float * coefs,
            float * deno);
        void resizeYmain(
            intptr_t srcSt, const uint8_t * src,
            intptr_t dstW, float * dst,
            intptr_t srcOY,
            const float * coefs);
        void resizeX(const float * src, uint8_t * dst);
        void resizeXborder(
            const float * src, uint8_t * dst,
            intptr_t begin, intptr_t end);
        void resizeXmain(
            const float * src, uint8_t * dst,
            intptr_t begin, intptr_t end);

        enum {
            //! for SIMD
            kVecStep  = 16, //!< __m256 x 2 
        };
        intptr_t m_SrcW;
        intptr_t m_SrcH;
        intptr_t m_DstW;
        intptr_t m_DstH;
        intptr_t m_NumCoefsX;
        intptr_t m_NumCoefsY;
        intptr_t m_NumTablesX;
        intptr_t m_NumTablesY;
        std::vector<float> m_TablesX;   //!< Lanczos table * m_NumTablesX (interleaved)
        std::vector<float> m_TablesY;   //!< Lanczos table * m_NumTablesY
        std::vector<int32_t> m_IndicesX;
        std::vector<float> m_Work;
        std::vector<float> m_Deno;
    };

    LanczosResizer::LanczosResizer(
        unsigned int degree,
        size_t srcW,
        size_t srcH,
        size_t dstW,
        size_t dstH,
        size_t pxScale)
    {
        m_Impl = new Impl(degree, srcW, srcH, dstW, dstH, pxScale);
    }

    LanczosResizer::~LanczosResizer()
    {
        delete m_Impl;
    }

    void LanczosResizer::resize(size_t srcSt, const unsigned char * src, size_t dstSt, unsigned char * dst)
    {
        m_Impl->resize(srcSt, src, dstSt, dst);
    }


    //==================================================
    // LanczosResizer::Impl
    //==================================================

    // Constructor
    LanczosResizer::Impl::Impl(unsigned int degree, size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t pxScale)
    {
        m_SrcW = srcW;
        m_SrcH = srcH;
        m_DstW = dstW;
        m_DstH = dstH;

        // setup coefficients
        if ( m_SrcW <= m_DstW ) {
            // horizontal: up-sampling
            m_NumCoefsX = 2 * degree;
        } else {
            // vertical: down-sampling
            // n = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            m_NumCoefsX = 2 * intptr_t(std::ceil((degree2 * m_SrcW) / double(m_DstW)));
        }
        if ( m_SrcH <= m_DstH ) {
            // vertical: up-sampling
            m_NumCoefsY = 2 * degree;
        } else {
            // vertical: down-sampling
            // n = 2*degree / scale
            size_t degree2 = std::max<size_t>(1, degree / pxScale);
            m_NumCoefsY = 2 * intptr_t(std::ceil((degree2 * m_SrcH) / double(m_DstH)));
        }
        intptr_t numTablesX = m_DstW / gcd(m_SrcW, m_DstW);
        m_NumTablesX = lcm(numTablesX, kVecStep);
        m_NumTablesY = m_DstH / gcd(m_SrcH, m_DstH);
        m_TablesX.reserve(m_NumTablesX * m_NumCoefsX);
        m_TablesX.resize(m_NumTablesX * m_NumCoefsX);
        m_TablesY.reserve(m_NumCoefsY * m_NumTablesY);
        m_TablesY.resize(m_NumCoefsY * m_NumTablesY);

        std::vector<float> tablesX(m_NumCoefsX * numTablesX);
        for ( intptr_t dstX = 0; dstX < numTablesX; ++dstX ) {
            float * table = &tablesX[dstX * m_NumCoefsX];
            double sumCoefs = setLanczosTable(degree, m_SrcW, m_DstW, dstX, pxScale, m_NumCoefsX, table);
            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                table[i] /= sumCoefs;
            }
        }
        // transpose and unroll X coefs
        //
        //   tablesX: A0A1A2A3
        //            B0B1B2B3
        //            C0C1C2C3
        //
        // m_TablesX: A0B0C0A0 B0C0A0B0 C0A0B0C0 (align to SIMD unit)
        //                      :
        //            A3B3C3A3 B3C3A3B3 C3A3B3C3
        for ( intptr_t iCoef = 0; iCoef < m_NumCoefsX; ++iCoef ) {
            for ( intptr_t dstX = 0; dstX < m_NumTablesX; ++dstX ) {
                intptr_t i = dstX % numTablesX * m_NumCoefsX + iCoef;
                m_TablesX[iCoef * m_NumTablesX + dstX] = tablesX[i];
            }
        }

        for ( intptr_t dstY = 0; dstY < m_NumTablesY; ++dstY ) {
            float * table = &m_TablesY[dstY * m_NumCoefsY];
            double sumCoefs = setLanczosTable(degree, m_SrcH, m_DstH, dstY, pxScale, m_NumCoefsY, table);
            for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
                table[i] /= sumCoefs;
            }
        }

        // allocate workspace
        m_Work.reserve(m_SrcW * getNumberOfProcs());
        m_Work.resize(m_SrcW * getNumberOfProcs());
        size_t maxW = std::max(m_SrcW, m_DstW);
        m_Deno.reserve(maxW * getNumberOfProcs());
        m_Deno.resize(maxW * getNumberOfProcs());

        // calc indices
        m_IndicesX.reserve(m_DstW);
        m_IndicesX.resize(m_DstW);
        for ( intptr_t dstX = 0; dstX < m_DstW; ++dstX ) {
            //       srcOX = floor(dstX / scale)
            intptr_t srcOX = dstX * m_SrcW / m_DstW + 1;
            m_IndicesX[dstX] = srcOX;
        }
    }

    void LanczosResizer::Impl::resize(size_t srcSt, const uint8_t * src, size_t dstSt, uint8_t * __restrict dst)
    {
        // resize
        if ( m_SrcH == m_DstH ) {
            float * work = &m_Work[0];
            for ( intptr_t y = 0; y < m_SrcH; ++y ) {
                for ( intptr_t x = 0; x < m_SrcW; ++x ) {
                    work[x] = src[srcSt * y + x];
                }
                resizeX(work, &dst[dstSt * y]);
            }
        } else {
            // vertical
            intptr_t numCoefsOn2 = m_NumCoefsY / 2;
            // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstH / double(m_SrcH))
            intptr_t mainBegin = ((numCoefsOn2 - 1) * m_DstH + m_SrcH-1) / m_SrcH;
            intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcH - numCoefsOn2) * m_DstH / m_SrcH);
            const float * tablesY = &m_TablesY[0];

#pragma omp parallel for
            for ( intptr_t dstY = 0; dstY < mainBegin; ++dstY ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                float * deno = &m_Deno[getThreadNumber() * m_SrcW];
                intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
                const float * coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                resizeYborder(
                    srcSt, &src[0],
                    m_SrcW, work,
                    srcOY,
                    coefs,
                    deno);
                resizeX(work, &dst[dstSt * dstY]);
            }
#pragma omp parallel for
            for ( intptr_t dstY = mainBegin; dstY < mainEnd; ++dstY ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
                const float * coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                resizeYmain(
                    srcSt, &src[0],
                    m_SrcW, work,
                    srcOY,
                    coefs);
                resizeX(work, &dst[dstSt * dstY]);
            }
#pragma omp parallel for
            for ( intptr_t dstY = mainEnd; dstY < m_DstH; ++dstY ) {
                float * work = &m_Work[getThreadNumber() * m_SrcW];
                float * deno = &m_Deno[getThreadNumber() * m_SrcW];
                intptr_t srcOY = dstY * m_SrcH / m_DstH + 1;
                const float * coefs = &tablesY[dstY % m_NumTablesY * m_NumCoefsY];
                resizeYborder(
                    srcSt, &src[0],
                    m_SrcW, work,
                    srcOY,
                    coefs,
                    deno);
                resizeX(work, &dst[dstSt * dstY]);
            }
        }
    }

    void LanczosResizer::Impl::resizeYborder(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstW, float * __restrict dst,
        intptr_t srcOY,
        const float * coefs,
        float * __restrict deno)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        float * nume = dst;

        std::memset(nume, 0, dstW * sizeof(*nume));
        std::memset(deno, 0, dstW * sizeof(*deno));

        for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
            float coef = coefs[i];
            for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                
                if ( 0 <= srcY && srcY < m_SrcH ) {
                    nume[dstX] += src[dstX + srcSt * srcY] * coef;
                    deno[dstX] += coef;
                }
            }
        }
        for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
            nume[dstX] /= deno[dstX];
        }
    }

    void LanczosResizer::Impl::resizeYmain(
        intptr_t srcSt, const uint8_t * src,
        intptr_t dstW, float * __restrict dst,
        intptr_t srcOY,
        const float * coefs)
    {
        intptr_t numCoefsOn2 = m_NumCoefsY / 2;
        float * nume = dst;

        std::memset(nume, 0, dstW * sizeof(*nume));

        for ( intptr_t i = 0; i < m_NumCoefsY; ++i ) {
            float coef = coefs[i];
            for ( intptr_t dstX = 0; dstX < dstW; ++dstX ) {
                intptr_t srcY = srcOY - numCoefsOn2 + i;
                nume[dstX] += src[dstX + srcSt * srcY] * coef;
            }
        }
    }

    void LanczosResizer::Impl::resizeX(const float * src, uint8_t * dst)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        // mainBegin = std::ceil((numCoefsOn2 - 1) * m_DstW / double(m_SrcW))
        intptr_t mainBegin = alignCeil(((numCoefsOn2 - 1) * m_DstW + m_SrcW-1) / m_SrcW, kVecStep);
        intptr_t mainEnd = std::max<intptr_t>(0, (m_SrcW - numCoefsOn2) * m_DstW / m_SrcW);
        intptr_t mainLen = alignFloor(mainEnd - mainBegin, kVecStep);
        mainEnd = mainBegin + mainLen;

        resizeXborder(src, dst, 0, mainBegin);
        resizeXmain(src, dst, mainBegin, mainEnd);
        resizeXborder(src, dst, mainEnd, m_DstW);
    }


    void LanczosResizer::Impl::resizeXborder(
        const float * src, uint8_t * dst,
        intptr_t begin, intptr_t end)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * tablesX = &m_TablesX[0];
        int32_t * indices = &m_IndicesX[0];

        intptr_t coefBegin = begin % m_NumTablesX;
        intptr_t iCoef = coefBegin;
        for ( intptr_t dstX = begin; dstX < end; ++dstX ) {
            //       srcOX = floor(dstX / scale) + 1;
            intptr_t srcOX = indices[dstX];
            float f32Nume  = 0;
            float f32Deno  = 0;

            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                const float * coefs = &tablesX[i * m_NumTablesX];
                intptr_t srcX = srcOX - numCoefsOn2 + i;
                if ( 0 <= srcX && srcX < m_SrcW ) {
                    float f32Coef = coefs[iCoef];
                    f32Nume += src[srcX] * f32Coef;
                    f32Deno += f32Coef;
                }
            }

            // iCoef = dstX % m_NumTablesX;
            iCoef++;
            if ( iCoef == m_NumTablesX ) {
                iCoef = 0;
            }

            dst[dstX] = clamp<int>(0, 255, round(f32Nume / f32Deno));
        }
    }

    void LanczosResizer::Impl::resizeXmain(
        const float * src, uint8_t * __restrict dst,
        intptr_t begin, intptr_t end)
    {
        intptr_t numCoefsOn2 = m_NumCoefsX / 2;
        const float * tablesX = &m_TablesX[0];
        int32_t * indices = &m_IndicesX[0];

        intptr_t coefBegin = begin % m_NumTablesX;
        intptr_t iCoef = coefBegin;
        for ( intptr_t dstX = begin; dstX < end; dstX += kVecStep ) {
            //      nume   = 0;
            __m256  vNume0 = _mm256_setzero_ps();
            __m256  vNume8 = _mm256_setzero_ps();
            //       srcX = floor(dstX / scale) - numCoefsOn2 + 1 + i;
            __m256i vSrcOX0 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 0]);
            __m256i vSrcOX8 = _mm256_loadu_si256((const __m256i*)&indices[dstX + 8]);

            for ( intptr_t i = 0; i < m_NumCoefsX; ++i ) {
                const float * coefs = &tablesX[i * m_NumTablesX];
                __m256i vOffset   = _mm256_set1_epi32(i - numCoefsOn2);

                //intptr_t srcX = indices[dstX + j] + offset;
                __m256i vSrcX0 = _mm256_add_epi32(vSrcOX0, vOffset);
                __m256i vSrcX8 = _mm256_add_epi32(vSrcOX8, vOffset);

                //nume[dstX + j] += src[srcX] * coefs[dstX + j];
                __m256  vSrc0    = _mm256_i32gather_ps(src, vSrcX0, sizeof(float));
                __m256  vSrc8    = _mm256_i32gather_ps(src, vSrcX8, sizeof(float));
                __m256  vCoefs0  = _mm256_loadu_ps(&coefs[iCoef + 0]);
                __m256  vCoefs8  = _mm256_loadu_ps(&coefs[iCoef + 8]);
                vNume0 = _mm256_fmadd_ps(vSrc0, vCoefs0, vNume0);
                vNume8 = _mm256_fmadd_ps(vSrc8, vCoefs8, vNume8);
            }

            __m128i vNume = cvtrps_epu8(vNume0, vNume8);
            _mm_storeu_si128((__m128i*)&dst[dstX], vNume);

            // iCoef = dstX % m_NumTablesX;
            iCoef += kVecStep;
            if ( iCoef == m_NumTablesX ) {
                iCoef = 0;
            }
        }
    }


}
