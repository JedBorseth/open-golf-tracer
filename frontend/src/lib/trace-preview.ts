import type { TraceAdjustments, TraceData, TracePoint } from './api'

export const DEFAULT_TRACE_ADJUSTMENTS: TraceAdjustments = {
  x_offset_px: 0,
  y_offset_px: 0,
  arc_scale: 1,
}

export function adjustedBallFlight(
  trace: TraceData,
  adjustments: TraceAdjustments,
): TracePoint[] {
  const address = adjustedPoint(trace.ball_address, trace.ball_address, adjustments)
  const flight = trace.ball_flight.map((point) =>
    adjustedPoint(point, trace.ball_address, adjustments),
  )
  if (trace.swing.impact_frame_index === null) {
    return flight
  }
  return [
    {
      ...address,
      frame_index: trace.swing.impact_frame_index,
    },
    ...flight,
  ]
}

export function flightPath(points: TracePoint[]): string {
  if (points.length === 0) {
    return ''
  }
  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`)
    .join(' ')
}

function adjustedPoint(
  point: TracePoint,
  baseAddress: TracePoint,
  adjustments: TraceAdjustments,
): TracePoint {
  return {
    ...point,
    x: point.x + adjustments.x_offset_px,
    y:
      baseAddress.y +
      adjustments.y_offset_px +
      (point.y - baseAddress.y) * adjustments.arc_scale,
  }
}
