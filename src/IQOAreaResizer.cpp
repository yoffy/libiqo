#include "IQOAreaResizerImpl.hpp"
#include "IQOHWCap.hpp"


namespace iqo {

    AreaResizer::AreaResizer(
        size_t srcW,
        size_t srcH,
        size_t dstW,
        size_t dstH
    ) {
        HWCap cap;

#if defined(IQO_CPU_X86)
        if ( cap.hasAVX512() && (m_Impl = AreaResizerImpl_new<ArchAVX512>()) ) {
            goto L_found;
        }
        if ( cap.hasAVX2FMA() && (m_Impl = AreaResizerImpl_new<ArchAVX2FMA>()) ) {
            goto L_found;
        }
        if ( cap.hasSSE4_1() && (m_Impl = AreaResizerImpl_new<ArchSSE4_1>()) ) {
            goto L_found;
        }
#elif defined(IQO_CPU_ARM)
        if ( cap.hasNEON() && (m_Impl = AreaResizerImpl_new<ArchNEON>()) ) {
            goto L_found;
        }
#endif

        m_Impl = AreaResizerImpl_new<ArchGeneric>();

L_found:
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
