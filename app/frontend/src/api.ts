import axios from 'axios'
import type {
  HealthResponse,
  DdgRequest,
  DdgResponse,
  ScanRequest,
  ScanResponse,
  ScanStreamStart,
  ScanStreamChunk,
  EpistasisRequest,
  EpistasisResponse,
  EpistasisStreamStart,
  EpistasisStreamChunk,
  StreamError,
  IdrRequest,
  IdrResponse,
  CompareRequest,
  CompareResponse,
  ExportRequest,
  HistoryItem,
  HistoryRecord,
} from './types'

const client = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

export interface StreamHandlers<TStart, TChunk, TResult> {
  onStart?: (data: TStart) => void
  onChunk?: (data: TChunk) => void
  onResult?: (data: TResult) => void
  onError?: (err: StreamError) => void
  signal?: AbortSignal
}

export type ScanStreamHandlers = StreamHandlers<
  ScanStreamStart,
  ScanStreamChunk,
  ScanResponse
>

export type EpistasisStreamHandlers = StreamHandlers<
  EpistasisStreamStart,
  EpistasisStreamChunk,
  EpistasisResponse
>

export async function getHealth(): Promise<HealthResponse> {
  const res = await client.get<HealthResponse>('/health')
  return res.data
}

export async function postDdg(payload: DdgRequest): Promise<DdgResponse> {
  const res = await client.post<DdgResponse>('/ddg', payload)
  return res.data
}

export async function postScan(payload: ScanRequest): Promise<ScanResponse> {
  const res = await client.post<ScanResponse>('/scan', payload)
  return res.data
}

export async function postEpistasis(payload: EpistasisRequest): Promise<EpistasisResponse> {
  const res = await client.post<EpistasisResponse>('/epistasis', payload)
  return res.data
}

export async function postIdr(payload: IdrRequest): Promise<IdrResponse> {
  const res = await client.post<IdrResponse>('/idr', payload)
  return res.data
}

export async function postCompare(payload: CompareRequest): Promise<CompareResponse> {
  const res = await client.post<CompareResponse>('/compare', payload)
  return res.data
}

/**
 * Internal helper for POST text/event-stream endpoints (/api/scan/stream, /api/epistasis/stream).
 * Surfaces pre-stream HTTP refusals (400 / 503) with their {detail} message and parses
 * potentially multi-chunk SSE frames cleanly.
 */
async function postSseStream<TStart, TChunk, TResult>(
  path: string,
  payload: unknown,
  handlers?: StreamHandlers<TStart, TChunk, TResult>,
  signal?: AbortSignal
): Promise<TResult> {
  const abortSignal = signal ?? handlers?.signal
  const res = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: abortSignal,
  })

  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const text = await res.text()
      if (text) {
        try {
          const parsed = JSON.parse(text) as { detail?: string }
          detail = parsed.detail || text
        } catch {
          detail = text
        }
      }
    } catch {
      // keep default status message
    }
    const errObj: StreamError = { status: res.status, detail }
    handlers?.onError?.(errObj)
    throw new Error(detail)
  }

  if (!res.body) {
    const detail = 'Streaming response body is unavailable in this browser'
    handlers?.onError?.({ status: 500, detail })
    throw new Error(detail)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalResult: TResult | null = null

  const processFrame = (rawFrame: string) => {
    let eventName = 'message'
    const dataLines: string[] = []
    for (const line of rawFrame.split('\n')) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim())
      }
    }
    if (dataLines.length === 0) return
    const parsedData = JSON.parse(dataLines.join('\n'))
    if (eventName === 'start') {
      handlers?.onStart?.(parsedData as TStart)
    } else if (eventName === 'chunk') {
      handlers?.onChunk?.(parsedData as TChunk)
    } else if (eventName === 'result') {
      finalResult = parsedData as TResult
      handlers?.onResult?.(finalResult)
    } else if (eventName === 'error') {
      const streamErr = parsedData as StreamError
      handlers?.onError?.(streamErr)
      throw new Error(streamErr.detail || `Stream failed (${streamErr.status})`)
    }
  }

  while (true) {
    const { value, done } = await reader.read()
    if (value) {
      buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n')
    } else if (done) {
      buffer += decoder.decode().replace(/\r\n/g, '\n')
    }

    let sepIdx: number
    while ((sepIdx = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sepIdx)
      buffer = buffer.slice(sepIdx + 2)
      if (frame.trim()) {
        processFrame(frame)
      }
    }

    if (done) {
      if (buffer.trim()) {
        processFrame(buffer)
      }
      break
    }
  }

  if (finalResult === null) {
    throw new Error('Stream ended before emitting a final result event')
  }
  return finalResult
}

export async function streamScan(
  payload: ScanRequest,
  handlers?: ScanStreamHandlers,
  signal?: AbortSignal
): Promise<ScanResponse> {
  return postSseStream<ScanStreamStart, ScanStreamChunk, ScanResponse>(
    '/scan/stream',
    payload,
    handlers,
    signal
  )
}

export async function streamEpistasis(
  payload: EpistasisRequest,
  handlers?: EpistasisStreamHandlers,
  signal?: AbortSignal
): Promise<EpistasisResponse> {
  return postSseStream<EpistasisStreamStart, EpistasisStreamChunk, EpistasisResponse>(
    '/epistasis/stream',
    payload,
    handlers,
    signal
  )
}

/**
 * POST /api/export, receive a blob, and trigger a browser download.
 */
export async function downloadExport(
  format: 'csv' | 'json' | 'pdb',
  data: unknown,
  filename?: string
): Promise<void> {
  const payload: ExportRequest = { format, data, filename }
  let res
  try {
    res = await client.post('/export', payload, { responseType: 'blob' })
  } catch (err: any) {
    // With responseType 'blob', error bodies arrive as a Blob — read it back
    // so the caller sees the real {detail: ...} message instead of "[object Blob]".
    if (err?.response?.data instanceof Blob) {
      const text = await err.response.data.text()
      try {
        throw new Error(JSON.parse(text).detail || text)
      } catch {
        throw new Error(text || 'Export failed')
      }
    }
    throw err
  }

  // Try to get filename from Content-Disposition header, else fall back
  const disposition: string = res.headers['content-disposition'] ?? ''
  let dlName = filename ?? `holo_geon_export.${format}`
  const match = disposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/)
  if (match?.[1]) {
    dlName = match[1].replace(/['"]/g, '')
  }

  const url = URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  a.download = dlName
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export async function getHistory(kind?: string, limit = 50): Promise<HistoryItem[]> {
  const res = await client.get<{ items: HistoryItem[] }>('/history', { params: { kind, limit } })
  return res.data.items
}

export async function getHistoryItem(id: number): Promise<HistoryRecord> {
  const res = await client.get<HistoryRecord>(`/history/${id}`)
  return res.data
}

export async function deleteHistoryItem(id: number): Promise<void> {
  await client.delete(`/history/${id}`)
}

export async function clearHistory(kind?: string): Promise<void> {
  await client.delete('/history', { params: { kind } })
}
