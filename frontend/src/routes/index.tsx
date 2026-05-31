import { createFileRoute } from '@tanstack/react-router'
import * as React from 'react'
import { createJob, getJob, getResultUrl, type JobResponse } from '~/lib/api'

export const Route = createFileRoute('/')({
  component: Home,
})

function Home() {
  const [file, setFile] = React.useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null)
  const [job, setJob] = React.useState<JobResponse | null>(null)
  const [isUploading, setIsUploading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  React.useEffect(() => {
    if (!job || job.status === 'complete' || job.status === 'failed') {
      return
    }

    const interval = window.setInterval(() => {
      void getJob(job.job_id)
        .then(setJob)
        .catch((pollError: Error) => setError(pollError.message))
    }, 1500)

    return () => window.clearInterval(interval)
  }, [job])

  async function upload() {
    if (!file) {
      setError('Choose an image or video first.')
      return
    }

    setIsUploading(true)
    setError(null)
    setJob(null)

    try {
      setJob(await createJob(file))
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Upload failed.')
    } finally {
      setIsUploading(false)
    }
  }

  const resultUrl = job ? getResultUrl(job) : null
  const isVideo = job?.media_kind === 'video'

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">Golf Tracer MVP</p>
        <h1>Upload a shot. Get a tracer back.</h1>
        <p className="lede">
          Start with an image or video from your phone. The backend queues the
          job, runs YOLO detection, tracks the ball, and renders the result.
        </p>
      </section>

      <section className="card">
        <label className="drop-zone">
          <span>Choose image or video</span>
          <small>JPEG, PNG, MOV, or MP4</small>
          <input
            accept="image/*,video/*"
            type="file"
            onChange={(event) => {
              setFile(event.currentTarget.files?.[0] ?? null)
              setJob(null)
              setError(null)
            }}
          />
        </label>

        {file ? (
          <div className="file-meta">
            <strong>{file.name}</strong>
            <span>{formatBytes(file.size)}</span>
          </div>
        ) : null}

        {previewUrl ? (
          file?.type.startsWith('video/') ? (
            <video className="preview" src={previewUrl} controls playsInline />
          ) : (
            <img className="preview" src={previewUrl} alt="Selected upload preview" />
          )
        ) : null}

        <button disabled={!file || isUploading} onClick={() => void upload()}>
          {isUploading ? 'Uploading...' : 'Start tracer job'}
        </button>
      </section>

      {job ? (
        <section className="card status-card">
          <div>
            <p className="eyebrow">Job Status</p>
            <h2>{job.status}</h2>
          </div>
          <code>{job.job_id}</code>
          {job.status === 'failed' ? (
            <p className="error">
              {job.error_code}: {job.error_message}
            </p>
          ) : null}
          {resultUrl ? (
            <div className="result">
              {isVideo ? (
                <video className="preview" src={resultUrl} controls playsInline />
              ) : (
                <img className="preview" src={resultUrl} alt="Processed tracer result" />
              )}
              <a href={resultUrl} download>
                Download result
              </a>
            </div>
          ) : null}
        </section>
      ) : null}

      {error ? <p className="error">{error}</p> : null}
    </main>
  )
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
