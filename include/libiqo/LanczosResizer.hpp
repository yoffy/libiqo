#pragma once

//! @file
//! @brief Lanczos image resampler

#include <stddef.h>

namespace iqo {

    //! Forward declaration
    class ILanczosResizerImpl;

    //! Lanczos image resampler
    class LanczosResizer
    {
    public:
        //! @brief Constructor
        //! @param degree   Window size of Lanczos (ex. A=2 means Lanczos2)
        //! @param srcW     Width of source image
        //! @param srcH     Height of source image
        //! @param dstW     Width of destination image
        //! @param dstH     Height of destination image
        //! @param pxScale  Scale of a pixel (ex. 2 when U plane of YUV420 image)
        //!
        //! Construct coefficients table.
        LanczosResizer(
            unsigned int degree,
            size_t srcW,
            size_t srcH,
            size_t dstW,
            size_t dstH,
            size_t pxScale=1
        );

        //! @brief Destructor
        ~LanczosResizer();

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
        LanczosResizer(const LanczosResizer &);
        LanczosResizer & operator=(const LanczosResizer &);

        ILanczosResizerImpl * m_Impl;
    };

}
