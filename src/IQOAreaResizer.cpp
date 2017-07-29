#include "IQOAreaResizerImpl.hpp"


namespace iqo {

    AreaResizer::AreaResizer(
        size_t srcW,
        size_t srcH,
        size_t dstW,
        size_t dstH
    ) {
#if defined(IQO_CPU_X86)
        if ( AreaResizerImpl_hasFeature<ArchSSE4_1>() ) {
            m_Impl = AreaResizerImpl_new<ArchSSE4_1>();
        } else
#endif
        {
            m_Impl = AreaResizerImpl_new<ArchGeneric>();
        }

        m_Impl->init(srcW, srcH, dstW, dstH);
    }

    AreaResizer::~AreaResizer()
    {
        delete m_Impl;
    }

    void AreaResizer::resize(
        size_t srcSt, const unsigned char * src,
        size_t dstSt, unsigned char * dst
    ) {
        m_Impl->resize(srcSt, src, dstSt, dst);
    }

}
