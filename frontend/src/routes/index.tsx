import { createFileRoute } from '@tanstack/react-router'
import * as React from 'react'
import {
  createJob,
  getJob,
  getResultUrl,
  getSourceUrl,
  getTrace,
  renderAdjustedJob,
  type JobResponse,
  type TraceAdjustments,
  type TraceData,
} from '~/lib/api'
import {
  DEFAULT_TRACE_ADJUSTMENTS,
  adjustedBallFlight,
  flightPath,
} from '~/lib/trace-preview'

export const Route = createFileRoute('/')({
  component: Home,
})

function Home() {
  const [file, setFile] = React.useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null)
  const [job, setJob] = React.useState<JobResponse | null>(null)
  const [trace, setTrace] = React.useState<TraceData | null>(null)
  const [adjustments, setAdjustments] = React.useState<TraceAdjustments>(
    DEFAULT_TRACE_ADJUSTMENTS,
  )
  const [isUploading, setIsUploading] = React.useState(false)
  const [isRendering, setIsRendering] = React.useState(false)
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

  React.useEffect(() => {
    if (!job || job.status !== 'complete' || !job.trace_url || job.media_kind !== 'video') {
      return
    }

    void getTrace(job)
      .then(setTrace)
      .catch((traceError: Error) => setError(traceError.message))
  }, [job])

  async function upload() {
    if (!file) {
      setError('Choose an image or video first.')
      return
    }

    setIsUploading(true)
    setError(null)
    setJob(null)
    setTrace(null)
    setAdjustments(DEFAULT_TRACE_ADJUSTMENTS)

    try {
      setJob(await createJob(file))
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Upload failed.')
    } finally {
      setIsUploading(false)
    }
  }

  async function rerender() {
    if (!job) {
      return
    }

    setIsRendering(true)
    setError(null)
    try {
      setJob(await renderAdjustedJob(job.job_id, adjustments))
    } catch (renderError) {
      setError(renderError instanceof Error ? renderError.message : 'Render failed.')
    } finally {
      setIsRendering(false)
    }
  }

  const resultUrl = job ? getResultUrl(job) : null
  const sourceUrl = job ? getSourceUrl(job) : null
  const isVideo = job?.media_kind === 'video'
  const adjustedFlight = trace ? adjustedBallFlight(trace, adjustments) : []
  const adjustedPath = flightPath(adjustedFlight)

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">Golf Swing Tracer</p>
        <h1>Upload a swing. See the club path and ball flight.</h1>
        <p className="lede">
          Upload a down-the-line or face-on swing video. We find the ball at
          address, track the club through impact, infer the launch path, and
          anchor the tracer to the scene as the camera moves.
        </p>
      </section>

      <section className="card">
        <label className="drop-zone">
          <span>Choose swing video (recommended)</span>
          <small>MP4 or MOV — JPEG/PNG supported with limited output</small>
          <input
            accept="image/*,video/*"
            type="file"
            onChange={(event) => {
              setFile(event.currentTarget.files?.[0] ?? null)
              setJob(null)
              setTrace(null)
              setAdjustments(DEFAULT_TRACE_ADJUSTMENTS)
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
          {isUploading ? 'Uploading...' : 'Trace swing and flight'}
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
          {trace && sourceUrl && isVideo ? (
            <div className="trace-editor">
              <div className="trace-preview">
                <video className="preview" src={sourceUrl} controls playsInline muted />
                <svg
                  aria-label="Adjusted ball flight preview"
                  className="trace-overlay"
                  preserveAspectRatio="xMidYMid meet"
                  viewBox={`0 0 ${trace.video.width} ${trace.video.height}`}
                >
                  {adjustedPath ? (
                    <>
                      <path className="trace-flight" d={adjustedPath} />
                      <circle
                        className="trace-address"
                        cx={adjustedFlight[0]?.x ?? 0}
                        cy={adjustedFlight[0]?.y ?? 0}
                        r="7"
                      />
                    </>
                  ) : null}
                </svg>
              </div>

              <div className="slider-grid">
                <Slider
                  label="Horizontal placement"
                  max={220}
                  min={-220}
                  step={1}
                  value={adjustments.x_offset_px}
                  valueLabel={`${adjustments.x_offset_px.toFixed(0)} px`}
                  onChange={(value) =>
                    setAdjustments((current) => ({ ...current, x_offset_px: value }))
                  }
                />
                <Slider
                  label="Vertical placement"
                  max={220}
                  min={-220}
                  step={1}
                  value={adjustments.y_offset_px}
                  valueLabel={`${adjustments.y_offset_px.toFixed(0)} px`}
                  onChange={(value) =>
                    setAdjustments((current) => ({ ...current, y_offset_px: value }))
                  }
                />
                <Slider
                  label="Arc"
                  max={2.2}
                  min={0.35}
                  step={0.05}
                  value={adjustments.arc_scale}
                  valueLabel={`${adjustments.arc_scale.toFixed(2)}x`}
                  onChange={(value) =>
                    setAdjustments((current) => ({ ...current, arc_scale: value }))
                  }
                />
              </div>

              <div className="action-row">
                <button type="button" onClick={() => setAdjustments(DEFAULT_TRACE_ADJUSTMENTS)}>
                  Reset preview
                </button>
                <button
                  disabled={isRendering || job.status === 'running'}
                  type="button"
                  onClick={() => void rerender()}
                >
                  {isRendering || job.status === 'running' ? 'Rendering...' : 'Render adjusted video'}
                </button>
              </div>
            </div>
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

type SliderProps = {
  label: string
  max: number
  min: number
  step: number
  value: number
  valueLabel: string
  onChange: (value: number) => void
}

function Slider({ label, max, min, onChange, step, value, valueLabel }: SliderProps) {
  return (
    <label className="slider-control">
      <span>
        {label}
        <strong>{valueLabel}</strong>
      </span>
      <input
        max={max}
        min={min}
        step={step}
        type="range"
        value={value}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  )
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
