#include "IQOLinearResizerImpl.hpp"


namespace iqo {

    LinearResizer::LinearResizer(
        size_t srcW,
        size_t srcH,
        size_t dstW,
        size_t dstH
    ) {
#if defined(IQO_CPU_X86)
        if ( LinearResizerImpl_hasFeature<ArchAVX512>() ) {
            m_Impl = LinearResizerImpl_new<ArchAVX512>();
        } else if ( LinearResizerImpl_hasFeature<ArchAVX2FMA>() ) {
            m_Impl = LinearResizerImpl_new<ArchAVX2FMA>();
        } else if ( LinearResizerImpl_hasFeature<ArchSSE4_1>() ) {
            m_Impl = LinearResizerImpl_new<ArchSSE4_1>();
        } else
#elif defined(IQO_CPU_ARM)
        if ( LinearResizerImpl_hasFeature<ArchNEON>() ) {
            m_Impl = LinearResizerImpl_new<ArchNEON>();
        } else
#endif
        {
            m_Impl = LinearResizerImpl_new<ArchGeneric>();
        }

        m_Impl->init(srcW, srcH, dstW, dstH);
    }

    LinearResizer::~LinearResizer()
    {
        delete m_Impl;
    }

    void LinearResizer::resize(
        size_t srcSt, const unsigned char * src,
        size_t dstSt, unsigned char * dst
    ) {
        m_Impl->resize(srcSt, src, dstSt, dst);
    }

}
