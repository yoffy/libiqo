#include <stdint.h>
#include <cmath>
#include <algorithm>

#include "libiqo/LanczosResizer.hpp"
#include "libiqo/Types.hpp"

namespace iqo {

    class ILanczosResizerImpl
    {
    public:
        //! Destructor
        virtual ~ILanczosResizerImpl() {}

        //! Construct impl
        virtual void init(
            unsigned int degree,
            size_t srcW, size_t srcH,
            size_t dstW, size_t dstH,
            size_t pxScale
        ) = 0;

        //! Run image resizing
        virtual void resize(
            size_t srcSt, const unsigned char * src,
            size_t dstSt, unsigned char * dst
        ) = 0;
    };

    //! MUST NOT define inline function to inherited function.
    //! Because class method inline function will be weak symbol.
    template<class ARCH>
    class LanczosResizerImpl : public ILanczosResizerImpl
    {
    public:
        virtual ~LanczosResizerImpl() {}

        virtual void init(
            unsigned int degree,
            size_t srcW, size_t srcH,
            size_t dstW, size_t dstH,
            size_t pxScale
        ) {
            (void)degree;
            (void)srcW;
            (void)srcH;
            (void)dstW;
            (void)dstH;
            (void)pxScale;
        }

        virtual void resize(
            size_t srcSt, const unsigned char * src,
            size_t dstSt, unsigned char * dst
        ) {
            (void)srcSt;
            (void)src;
            (void)dstSt;
            (void)dst;
        }
    };

    //! Returns feature capability
    template<class ARCH>
    bool LanczosResizerImpl_hasFeature()
    {
        return false;
    }

    //! new LanczosResizerImpl<ARCH>
    template<class ARCH>
    ILanczosResizerImpl * LanczosResizerImpl_new()
    {
        return NULL;
    }

    template<> bool LanczosResizerImpl_hasFeature<ArchGeneric>();
    template<> bool LanczosResizerImpl_hasFeature<ArchSSE4_1>();
    template<> bool LanczosResizerImpl_hasFeature<ArchAVX2FMA>();
    template<> bool LanczosResizerImpl_hasFeature<ArchAVX512>();
    template<> bool LanczosResizerImpl_hasFeature<ArchNEON>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchGeneric>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchSSE4_1>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchAVX2FMA>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchAVX512>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchNEON>();


    //! Linear integer interpolation
    class LinearIterator
    {
    public:
        LinearIterator(ptrdiff_t dx, ptrdiff_t dy)
        {
            m_DX = dx;
            m_DY = dy;
            m_X = 0;
            m_Y = 0;
        }

        //! set x
        void setX(ptrdiff_t x)
        {
            m_X = (m_DY * x) % m_DX;
            m_Y = (m_DY * x) / m_DX;
        }

        //! get y
        ptrdiff_t operator*() const
        {
            return m_Y;
        }

        //! ++x
        LinearIterator & operator++()
        {
            advance();
            return *this;
        }

        //! x++
        LinearIterator operator++(int)
        {
            LinearIterator tmp(*this);
            advance();
            return tmp;
        }

    private:
        void advance()
        {
            m_X += m_DY;
            while ( m_X >= m_DX ) {
                ++m_Y;
                m_X -= m_DX;
            }
        }

        ptrdiff_t m_DX;
        ptrdiff_t m_DY;
        ptrdiff_t m_X;
        ptrdiff_t m_Y;
    };

    //! Calculate number of coefficients for Lanczos resampling
    size_t calcNumCoefsForLanczos(int degree, size_t srcLen, size_t dstLen, size_t pxScale);

    //! @brief Set Lanczos table
    //! @param degree     Window size of Lanczos (ex. A=2 means Lanczos2)
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param dstOffset  The coordinate of the destination image
    //! @param pxScale    Scale of a pixel (ex. 2 when U plane of YUV420 image)
    //! @param numCoefs   Size of table
    //! @param fTable     The table (float or double)
    //! @return Sum of the table
    //!
    //! Calculate Lanczos coefficients from `-degree` to `+degree`.
    //!
    //! tableLen should be `2*degree` when up sampling.
    float setLanczosTable(
        int degree,
        size_t srcLen,
        size_t dstLen,
        ptrdiff_t dstOffset,
        size_t pxScale,
        ptrdiff_t numCoefs,
        float * fTable
    );

}

namespace {

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

    template<typename T>
    static inline T alignFloor(T v, T alignment)
    {
        return v / alignment * alignment;
    }

    template<typename T>
    static inline T alignCeil(T v, T alignment)
    {
        return (v + (alignment - 1)) / alignment * alignment;
    }

    static inline ptrdiff_t gcd(ptrdiff_t a, ptrdiff_t b)
    {
        ptrdiff_t r = a % b;

        while ( r ) {
            a = b;
            b = r;
            r = a % b;
        }

        return b;
    }

    static inline int64_t lcm(int64_t a, int64_t b)
    {
        return a * b / gcd(a, b);
    }

}
