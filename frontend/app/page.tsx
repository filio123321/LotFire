"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { io, type Socket } from "socket.io-client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Upload, Camera, Link, Flame, Loader2, AlertTriangle, Play, Square } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface DetectionResult {
  type: "image" | "video"
  data: string | VideoDetection[]
  timestamp?: string
}

interface VideoDetection {
  timestamp: number
  detections: Array<{
    bbox: [number, number, number, number]
    confidence: number
    class: string
  }>
}

export default function FireDetectionApp() {
  const API_BASE = "http://172.20.10.7:8080"

  // Tabs & global state
  const [activeTab, setActiveTab] = useState("upload")
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [url, setUrl] = useState("")

  // Detection parameters
  const [confidence, setConfidence] = useState<[number]>([0.5])
  const [iou, setIou] = useState<[number]>([0.45])
  const [imageSize, setImageSize] = useState<[number]>([640])

  // File upload ref
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Stream state & refs
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [streamError, setStreamError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const socketRef = useRef<Socket | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopStream()
    }
  }, [])

  // Start the live WebSocket-based stream
  const startStream = async () => {
    setStreamError(null)
    setError(null)
    setIsStreamActive(true)

    try {
      // 1) Grab webcam
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      // 2) Connect socket.io
      socketRef.current = io(API_BASE, { transports: ["websocket"] })
      socketRef.current.on("annotated_frame", (bytes: ArrayBuffer) => {
        const blob = new Blob([bytes], { type: "image/jpeg" })
        if (imgRef.current) {
          imgRef.current.src = URL.createObjectURL(blob)
        }
      })
      socketRef.current.on("error", (err: { message: string }) => {
        setStreamError(err.message)
        stopStream()
      })

      // 3) Frame sender
      const sendFrame = () => {
        if (!videoRef.current || !socketRef.current) return
        const v = videoRef.current
        const c = document.createElement("canvas")
        c.width = v.videoWidth
        c.height = v.videoHeight
        const ctx = c.getContext("2d")!
        ctx.drawImage(v, 0, 0)
        c.toBlob(
          (blob) => {
            if (blob) {
              blob.arrayBuffer().then((buf) => {
                socketRef.current!.emit("frame", buf, { conf: confidence[0], iou: iou[0], imgsz: imageSize[0] })
              })
            }
          },
          "image/jpeg",
          0.8,
        )
      }

      // Initial send + interval
      sendFrame()
      intervalRef.current = setInterval(() => {
        sendFrame()
      }, 1000 * 3) // you can adjust or tie to a stateful `captureInterval`
    } catch (err: any) {
      setStreamError(err.message || "Failed to access webcam")
      stopStream()
    }
  }

  const stopStream = () => {
    setIsStreamActive(false)
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach((t) => t.stop())
      videoRef.current.srcObject = null
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (socketRef.current) {
      socketRef.current.disconnect()
      socketRef.current = null
    }
  }

  const clearResults = () => {
    stopStream()
    setResult(null)
    setError(null)
    setUrl("")
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const detectFromFile = async (file: File | Blob, type: "image" | "video") => {
    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append(type === "image" ? "image" : "video", file)
      formData.append("conf", confidence[0].toString())
      formData.append("iou", iou[0].toString())
      formData.append("imgsz", imageSize[0].toString())

      const endpoint = type === "image" ? "/detect/image" : "/detect/video"
      console.log("Sending request to:", `${API_BASE}${endpoint}`)

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
        mode: "cors",
      })

      console.log("Response status:", response.status)

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Detection failed: ${response.status} - ${errorText}`)
      }

      if (type === "image") {
        const blob = await response.blob()
        const imageUrl = URL.createObjectURL(blob)
        setResult({
          type: "image",
          data: imageUrl,
          timestamp: new Date().toLocaleString(),
        })
      } else {
        const jsonResult = await response.json()
        setResult({
          type: "video",
          data: jsonResult,
          timestamp: new Date().toLocaleString(),
        })
      }
    } catch (err) {
      console.error("Detection error:", err)
      setError(err instanceof Error ? err.message : "Detection failed")
    } finally {
      setIsLoading(false)
    }
  }

  const detectFromUrl = async () => {
    if (!url.trim()) {
      setError("Please enter a valid URL")
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      console.log("Sending URL request to:", `${API_BASE}/detect/url`)

      const response = await fetch(`${API_BASE}/detect/url`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: url.trim(),
          conf: confidence[0],
          iou: iou[0],
          imgsz: imageSize[0],
        }),
        mode: "cors",
      })

      console.log("Response status:", response.status)

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Detection failed: ${response.status} - ${errorText}`)
      }

      const blob = await response.blob()
      const imageUrl = URL.createObjectURL(blob)
      setResult({
        type: "image",
        data: imageUrl,
        timestamp: new Date().toLocaleString(),
      })
    } catch (err) {
      console.error("URL detection error:", err)
      setError(err instanceof Error ? err.message : "Detection failed")
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const isVideo = file.type.startsWith("video/")
    const isImage = file.type.startsWith("image/")

    if (!isVideo && !isImage) {
      setError("Please select an image or video file")
      return
    }

    detectFromFile(file, isVideo ? "video" : "image")
  }

  const renderVideoResults = (detections: VideoDetection[]) => {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Video Analysis Results</h3>
        <div className="max-h-96 overflow-y-auto space-y-2">
          {detections.map((detection, index) => (
            <Card key={index} className="p-3">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">Timestamp: {detection.timestamp}s</span>
                <span className="text-sm text-muted-foreground">{detection.detections.length} detection(s)</span>
              </div>
              {detection.detections.map((det, detIndex) => (
                <div key={detIndex} className="text-xs bg-muted p-2 rounded mb-1">
                  <div className="flex justify-between">
                    <span className="font-medium text-red-600">{det.class}</span>
                    <span>Confidence: {(det.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-muted-foreground">BBox: [{det.bbox.map((b) => b.toFixed(0)).join(", ")}]</div>
                </div>
              ))}
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-red-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Flame className="h-8 w-8 text-red-500" />
            <h1 className="text-4xl font-bold text-gray-900">Fire Detection System</h1>
          </div>
          <p className="text-lg text-gray-600">Upload images, videos, or use live camera stream to detect fire</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Card */}
          <Card className="h-fit">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Input Source
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="upload">Upload</TabsTrigger>
                  <TabsTrigger value="stream">Live Stream</TabsTrigger>
                  <TabsTrigger value="url">URL</TabsTrigger>
                </TabsList>

                {/* Upload Tab */}
                <TabsContent value="upload" className="space-y-4">
                  <div>
                    <Label htmlFor="file-upload">Select Image or Video</Label>
                    <Input
                      id="file-upload"
                      type="file"
                      accept="image/*,video/*"
                      onChange={handleFileUpload}
                      ref={fileInputRef}
                      className="mt-2"
                    />
                  </div>
                </TabsContent>

                {/* Live Stream Tab */}
                <TabsContent value="stream" className="space-y-4">
                  {!isStreamActive ? (
                    <div className="text-center space-y-4">
                      <div className="p-6 border-2 border-dashed border-gray-300 rounded-lg">
                        <Camera className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                        <p className="text-sm text-gray-600 mb-4">
                          Start live camera stream with real-time fire detection
                        </p>
                        <Button onClick={startStream}>
                          <Play className="h-4 w-4 mr-2" />
                          Start Live Stream
                        </Button>
                      </div>
                      {streamError && (
                        <Alert variant="destructive">
                          <AlertTriangle className="h-4 w-4" />
                          <AlertDescription>{streamError}</AlertDescription>
                        </Alert>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <video ref={videoRef} autoPlay muted className="w-full rounded-lg border" />
                      <div className="flex justify-center">
                        <Button variant="outline" onClick={stopStream}>
                          <Square className="h-4 w-4 mr-2" />
                          Stop Stream
                        </Button>
                      </div>
                    </div>
                  )}
                </TabsContent>

                {/* URL Tab */}
                <TabsContent value="url" className="space-y-4">
                  <div>
                    <Label htmlFor="image-url">Image URL</Label>
                    <Input
                      id="image-url"
                      type="url"
                      placeholder="https://example.com/image.jpg"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      className="mt-2"
                    />
                  </div>
                  <Button onClick={detectFromUrl} disabled={isLoading || !url.trim()}>
                    <Link className="h-4 w-4 mr-2" />
                    Detect from URL
                  </Button>
                </TabsContent>
              </Tabs>

              {/* Detection Parameters */}
              <div className="mt-6 space-y-4 border-t pt-4">
                <h3 className="font-semibold text-sm">Detection Parameters</h3>
                <p className="text-xs text-muted-foreground">
                  {isStreamActive ? "Changes will apply in real time" : "Configure detection sensitivity"}
                </p>
                <div>
                  <Label className="text-sm">Confidence: {confidence[0]}</Label>
                  <Slider
                    value={confidence}
                    onValueChange={(v) => setConfidence(v as [number])}
                    max={1}
                    min={0.1}
                    step={0.05}
                    className="mt-2"
                  />
                </div>
                <div>
                  <Label className="text-sm">IoU: {iou[0]}</Label>
                  <Slider
                    value={iou}
                    onValueChange={(v) => setIou(v as [number])}
                    max={1}
                    min={0.1}
                    step={0.05}
                    className="mt-2"
                  />
                </div>
                <div>
                  <Label className="text-sm">Image Size: {imageSize[0]}px</Label>
                  <Slider
                    value={imageSize}
                    onValueChange={(v) => setImageSize(v as [number])}
                    max={1280}
                    min={320}
                    step={32}
                    className="mt-2"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Results Card */}
          <Card className="h-fit">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Detection Results
                </CardTitle>
                <Button variant="outline" size="sm" onClick={clearResults}>
                  Clear
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {isLoading && (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin mr-2" />
                  <span>Analyzing for fire detection...</span>
                </div>
              )}
              {error && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              {isStreamActive && (
                <div className="space-y-4">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold">Live Detection Stream</h3>
                    <span className="text-sm text-green-600 font-medium">🔴 LIVE</span>
                  </div>
                  <img ref={imgRef} alt="Live detection stream" className="w-full rounded-lg border" />
                </div>
              )}
              {result &&
                !isLoading &&
                (result.type === "image" ? (
                  <div>
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold">Annotated Result</h3>
                      <span className="text-sm text-muted-foreground">{result.timestamp}</span>
                    </div>
                    <img
                      src={(result.data as string) || "/placeholder.svg"}
                      alt="Fire detection result"
                      className="w-full rounded-lg border shadow-sm"
                    />
                  </div>
                ) : (
                  renderVideoResults(result.data as VideoDetection[])
                ))}
              {!result && !isLoading && !error && !isStreamActive && (
                <div className="text-center py-12 text-muted-foreground">
                  <Flame className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Upload, live-stream, or URL to detect fire</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
