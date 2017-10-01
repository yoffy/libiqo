#pragma once

#include <stdint.h>

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

    //! new LanczosResizerImpl<ARCH>
    template<class ARCH>
    ILanczosResizerImpl * LanczosResizerImpl_new()
    {
        return NULL;
    }

    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchGeneric>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchSSE4_1>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchAVX2FMA>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchAVX512>();
    template<> ILanczosResizerImpl * LanczosResizerImpl_new<ArchNEON>();


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
