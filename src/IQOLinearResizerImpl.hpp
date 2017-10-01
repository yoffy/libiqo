#pragma once

#include <stdint.h>

#include "libiqo/LinearResizer.hpp"
#include "libiqo/Types.hpp"

namespace iqo {

    class ILinearResizerImpl
    {
    public:
        //! Destructor
        virtual ~ILinearResizerImpl() {}

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
    class LinearResizerImpl : public ILinearResizerImpl
    {
    public:
        virtual ~LinearResizerImpl() {}

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

    //! new LinearResizerImpl<ARCH>
    template<class ARCH>
    ILinearResizerImpl * LinearResizerImpl_new()
    {
        return NULL;
    }

    template<> ILinearResizerImpl * LinearResizerImpl_new<ArchGeneric>();
    template<> ILinearResizerImpl * LinearResizerImpl_new<ArchSSE4_1>();
    template<> ILinearResizerImpl * LinearResizerImpl_new<ArchAVX2FMA>();
    template<> ILinearResizerImpl * LinearResizerImpl_new<ArchAVX512>();
    template<> ILinearResizerImpl * LinearResizerImpl_new<ArchNEON>();


    //! @brief Set Linear table
    //! @param srcLen     Number of pixels of the source image
    //! @param dstLen     Number of pixels of the destination image
    //! @param fTable     The table (float or double)
    void setLinearTable(
        size_t srcLen,
        size_t dstLen,
        float * fTable
    );

}
