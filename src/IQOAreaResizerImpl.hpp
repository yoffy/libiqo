#pragma once

#include <stdint.h>

#include "libiqo/AreaResizer.hpp"
#include "libiqo/Types.hpp"

namespace iqo {

    class IAreaResizerImpl
    {
    public:
        //! Destructor
        virtual ~IAreaResizerImpl() {}

        //! Construct impl
        virtual void init(
            size_t srcW, size_t srcH,
            size_t dstW, size_t dstH
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
    class AreaResizerImpl : public IAreaResizerImpl
    {
    public:
        virtual ~AreaResizerImpl() {}

        virtual void init(
            size_t srcW, size_t srcH,
            size_t dstW, size_t dstH
        ) {
            (void)srcW;
            (void)srcH;
            (void)dstW;
            (void)dstH;
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

    //! new AreaResizerImpl<ARCH>
    template<class ARCH>
    IAreaResizerImpl * AreaResizerImpl_new()
    {
        return NULL;
    }

    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchGeneric>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchSSE4_1>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchAVX2FMA>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchAVX512>();
    template<> IAreaResizerImpl * AreaResizerImpl_new<ArchNEON>();


    //! Calculate number of coefficients for area resampling
    size_t calcNumCoefsForArea(size_t srcLen, size_t dstLen);

    //! @brief Set Area table
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param dstOffset  The coordinate of the destination image
    //! @param numCoefs   Size of table
    //! @param fTable     The table (float or double)
    //! @return Sum of the table
    float setAreaTable(
        size_t srcLen,
        size_t dstLen,
        ptrdiff_t dstOffset,
        ptrdiff_t numCoefs,
        float * fTable
    );

}
