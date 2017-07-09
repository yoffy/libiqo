#include "IQOLanczosResizerImpl.hpp"


namespace iqo {

    LanczosResizer::LanczosResizer(
        unsigned int degree,
        size_t srcW,
        size_t srcH,
        size_t dstW,
        size_t dstH,
        size_t pxScale
    ) {
        if ( LanczosResizerImpl_hasFeature<ArchAVX2FMA>() ) {
            m_Impl = LanczosResizerImpl_new<ArchAVX2FMA>();
        } else if ( LanczosResizerImpl_hasFeature<ArchSSE4_1>() ) {
            m_Impl = LanczosResizerImpl_new<ArchSSE4_1>();
        } else {
            m_Impl = LanczosResizerImpl_new<ArchGeneric>();
        }

        m_Impl->init(degree, srcW, srcH, dstW, dstH, pxScale);
    }

    LanczosResizer::~LanczosResizer()
    {
        delete m_Impl;
    }

    void LanczosResizer::resize(
        size_t srcSt, const unsigned char * src,
        size_t dstSt, unsigned char * dst
    ) {
        m_Impl->resize(srcSt, src, dstSt, dst);
    }

}
