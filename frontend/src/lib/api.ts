export type JobStatus = 'queued' | 'running' | 'complete' | 'failed'
export type MediaKind = 'image' | 'video'

export type JobResponse = {
  job_id: string
  status: JobStatus
  media_kind: MediaKind
  result_url: string | null
  error_code: string | null
  error_message: string | null
  created_at: string
  updated_at: string
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

export function getResultUrl(job: JobResponse): string | null {
  if (!job.result_url) {
    return null
  }
  return `${API_BASE_URL}${job.result_url}`
}

async function parseResponse(response: Response): Promise<JobResponse> {
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
