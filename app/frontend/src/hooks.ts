import { useCallback, useRef, useState } from 'react'

/**
 * Width of an element that may mount later (callback ref + ResizeObserver).
 * Used to size wide Plotly figures to their card instead of a fixed pixel width.
 */
export function useMeasuredWidth<T extends HTMLElement>(): [(el: T | null) => void, number] {
  const [width, setWidth] = useState(0)
  const observer = useRef<ResizeObserver | null>(null)

  const ref = useCallback((el: T | null) => {
    observer.current?.disconnect()
    observer.current = null
    if (!el) return
    setWidth(Math.floor(el.getBoundingClientRect().width))
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver((entries) => {
        const w = Math.floor(entries[0]?.contentRect.width ?? 0)
        if (w > 0) setWidth(w)
      })
      ro.observe(el)
      observer.current = ro
    }
  }, [])

  return [ref, width]
}
