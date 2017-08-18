#pragma once

//! @file
//! @brief Linear image resampler

#include <stddef.h>

namespace iqo {

    //! Forward declaration
    class ILinearResizerImpl;

    //! Linear image resampler
    class LinearResizer
    {
    public:
        //! @brief Constructor
        //! @param srcW     Width of source image
        //! @param srcH     Height of source image
        //! @param dstW     Width of destination image
        //! @param dstH     Height of destination image
        //!
        //! Construct coefficients table.
        LinearResizer(
            size_t srcW,
            size_t srcH,
            size_t dstW,
            size_t dstH
        );

        //! @brief Destructor
        ~LinearResizer();

        //! @brief Resize image
        //! @param srcSt  Stride of src (in byte)
        //! @param src    Source image
        //! @param dstSt  Stride of dst (in byte)
        //! @param dst    Destination image
        //!
        //! Size of src and dst has to be set in constructor.
        //!
        //! srcSt and dstSt are line length in byte.
        void resize(
            size_t srcSt,
            const unsigned char * src,
            size_t dstSt,
            unsigned char * dst
        );

    private:
        // no copy
        LinearResizer(const LinearResizer &);
        LinearResizer & operator=(const LinearResizer &);

        ILinearResizerImpl * m_Impl;
    };

}
