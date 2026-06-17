export type JobStatus = 'queued' | 'running' | 'complete' | 'failed'
export type MediaKind = 'image' | 'video'

export type JobResponse = {
  job_id: string
  status: JobStatus
  media_kind: MediaKind
  result_url: string | null
  source_url: string | null
  trace_url: string | null
  error_code: string | null
  error_message: string | null
  created_at: string
  updated_at: string
}

export type TracePoint = {
  frame_index: number
  x: number
  y: number
  confidence: number
  source: string
}

export type SwingTrace = {
  points: TracePoint[]
  impact_frame_index: number | null
  impact_x: number | null
  impact_y: number | null
}

export type TraceData = {
  video: {
    width: number
    height: number
    fps: number
    frame_count: number
  }
  swing: SwingTrace
  ball_address: TracePoint
  ball_flight: TracePoint[]
}

export type TraceAdjustments = {
  x_offset_px: number
  y_offset_px: number
  arc_scale: number
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export async function createJob(file: File): Promise<JobResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/api/jobs`, {
    method: 'POST',
    body: formData,
  })

  return parseResponse(response)
}

export async function getJob(jobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/api/jobs/${jobId}`)
  return parseResponse(response)
}

export async function getTrace(job: JobResponse): Promise<TraceData> {
  if (!job.trace_url) {
    throw new Error('Trace geometry is not ready yet.')
  }
  const response = await fetch(`${API_BASE_URL}${job.trace_url}`)
  return parseResponse(response)
}

export async function renderAdjustedJob(
  jobId: string,
  adjustments: TraceAdjustments,
): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/api/jobs/${jobId}/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(adjustments),
  })
  return parseResponse(response)
}

export function getResultUrl(job: JobResponse): string | null {
  if (!job.result_url) {
    return null
  }
  return `${API_BASE_URL}${job.result_url}?v=${encodeURIComponent(job.updated_at)}`
}

export function getSourceUrl(job: JobResponse): string | null {
  if (!job.source_url) {
    return null
  }
  return `${API_BASE_URL}${job.source_url}`
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return response.json()
  }

  let message = `Request failed with status ${response.status}`
  try {
    const body = await response.json()
    message = body.detail ?? message
  } catch {
    // Keep the status-based message.
  }

  throw new Error(message)
}
