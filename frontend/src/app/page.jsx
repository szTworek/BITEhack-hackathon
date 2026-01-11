"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { fetchApi } from "@/lib/api";

const MAPBOX_ACCESS_TOKEN = "";

// Function to create animated pulsing dot
const createPulsingDot = (map, size = 200) => {
  return {
    width: size,
    height: size,
    data: new Uint8Array(size * size * 4),
    context: null,

    onAdd: function () {
      const canvas = document.createElement("canvas");
      canvas.width = this.width;
      canvas.height = this.height;
      this.context = canvas.getContext("2d");
    },

    render: function () {
      const duration = 1000;
      const t = (performance.now() % duration) / duration;

      const radius = (size / 2) * 0.3;
      const outerRadius = (size / 2) * 0.7 * t + radius;
      const context = this.context;

      context.clearRect(0, 0, this.width, this.height);

      // Outer pulsing circle - #C1121F with transparency
      context.beginPath();
      context.arc(this.width / 2, this.height / 2, outerRadius, 0, Math.PI * 2);
      context.fillStyle = `rgba(193, 18, 31, ${1 - t})`;
      context.fill();

      // Inner solid dot - #C1121F
      context.beginPath();
      context.arc(this.width / 2, this.height / 2, radius, 0, Math.PI * 2);
      context.fillStyle = "#C1121F";
      context.strokeStyle = "white";
      context.lineWidth = 2 + 4 * (1 - t);
      context.fill();
      context.stroke();

      this.data = context.getImageData(0, 0, this.width, this.height).data;

      map.triggerRepaint();

      return true;
    }
  };
};

export default function Home() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const handlePointClickRef = useRef(null);

  // Calculate default dates (5 hours back)
  const getDefaultDates = () => {
    const now = new Date();
    const fiveHoursAgo = new Date(now.getTime() - 5 * 60 * 60 * 1000);
    return {
      start: fiveHoursAgo.toISOString().split("T")[0],
      end: now.toISOString().split("T")[0],
    };
  };

  const defaultDates = getDefaultDates();
  const todayString = new Date().toISOString().split("T")[0];

  // Time range states
  const [startDate, setStartDate] = useState(defaultDates.start);
  const [endDate, setEndDate] = useState(defaultDates.end);
  const [availableDays, setAvailableHours] = useState([]);
  const [selectedDayIndex, setSelectedHourIndex] = useState(0);
  const [pointsByDay, setPointsByHour] = useState({});
  const [isTimeRangeLoading, setIsTimeRangeLoading] = useState(false);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const simulationIntervalRef = useRef(null);

  // Upscale states
  const [isUpscaling, setIsUpscaling] = useState(false);
  const [isUpscaled, setIsUpscaled] = useState(false);

  const handlePointClick = useCallback(async (pointId) => {
    setIsLoading(true);
    setIsDialogOpen(true);
    try {
      const pointData = await fetchApi(`/points/${pointId}`);
      setSelectedPoint(pointData);
    } catch (error) {
      console.error("Error fetching point data:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Update ref with current handler
  handlePointClickRef.current = handlePointClick;

  const closeDialog = () => {
    setIsDialogOpen(false);
    setSelectedPoint(null);
    setIsUpscaled(false);
    setIsUpscaling(false);
  };

  // Handle upscale - simulate API call
  const handleUpscale = async () => {
    setIsUpscaling(true);
    // Simulate API call - wait 2 seconds
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsUpscaled(true);
    setIsUpscaling(false);
  };

  // Load data for time range
  const loadTimeRangeData = async () => {
    if (!startDate || !endDate) {
      alert("Please select start and end dates");
      return;
    }
    setIsTimeRangeLoading(true);
    try {
      const data = await fetchApi(`/points/timerange?start_date=${startDate}&end_date=${endDate}`);
      setAvailableHours(data.hours);
      setPointsByHour(data.points);
      setSelectedHourIndex(0);
    } catch (error) {
      console.error("Error loading time range data:", error);
    } finally {
      setIsTimeRangeLoading(false);
    }
  };

  // Simulation - advance every second
  const toggleSimulation = () => {
    if (isSimulationRunning) {
      clearInterval(simulationIntervalRef.current);
      simulationIntervalRef.current = null;
      setIsSimulationRunning(false);
    } else {
      setIsSimulationRunning(true);
      simulationIntervalRef.current = setInterval(() => {
        setSelectedHourIndex((prev) => {
          if (prev >= availableDays.length - 1) {
            return 0; // Loop back to start
          }
          return prev + 1;
        });
      }, 1000);
    }
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (simulationIntervalRef.current) {
        clearInterval(simulationIntervalRef.current);
      }
    };
  }, []);

  // Update map when hour changes
  useEffect(() => {
    if (availableDays.length > 0 && map.current && map.current.getSource("points-source")) {
      const currentDay = availableDays[selectedDayIndex];
      const points = pointsByDay[currentDay];
      if (points) {
        map.current.getSource("points-source").setData(points);
      }
    }
  }, [selectedDayIndex, availableDays, pointsByDay]);

  // Initialize map
  useEffect(() => {
    if (map.current) return;

    mapboxgl.accessToken = MAPBOX_ACCESS_TOKEN;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/navigation-night-v1",
      center: [0, 20],
      zoom: 2,
      projection: "mercator"
    });

    map.current.on("load", () => {
      // Set water color to match theme
      map.current.setPaintProperty("water", "fill-color", "#141E30");

      // Add animated pulsing dot image
      const pulsingDot = createPulsingDot(map.current, 80);
      map.current.addImage("pulsing-dot", pulsingDot, { pixelRatio: 2 });

      setMapLoaded(true);
    });

    return () => {
      map.current?.remove();
    };
  }, []);

  // Fetch and display points
  useEffect(() => {
    if (!mapLoaded || !map.current) return;

    const loadPoints = async () => {
      try {
        const geojsonData = await fetchApi("/points");

        // Add source with GeoJSON data
        map.current.addSource("points-source", {
          type: "geojson",
          data: geojsonData
        });

        // Add layer with pulsing points
        map.current.addLayer({
          id: "points-layer",
          type: "symbol",
          source: "points-source",
          layout: {
            "icon-image": "pulsing-dot",
            "icon-allow-overlap": true
          }
        });

        // Handle point click
        map.current.on("click", "points-layer", (e) => {
          if (e.features && e.features.length > 0) {
            const pointId = e.features[0].properties.id;
            handlePointClickRef.current?.(pointId);
          }
        });

        // Change cursor on hover
        map.current.on("mouseenter", "points-layer", () => {
          map.current.getCanvas().style.cursor = "pointer";
        });

        map.current.on("mouseleave", "points-layer", () => {
          map.current.getCanvas().style.cursor = "";
        });

      } catch (error) {
        console.error("Error fetching points:", error);
      }
    };

    loadPoints();
  }, [mapLoaded]);

  return (
    <div className="flex flex-col h-screen w-screen">
      {/* Top bar with logo */}
      <div className="bg-panel px-6 py-3 flex items-center justify-center">
        <div className="flex items-center gap-3 justify-center">
          <span className="text-text-primary text-xl font-semibold">DETECTING SHIPS BEYOND AIS</span>
        </div>
      </div>

      {/* Map - takes remaining space */}
      <div className="flex-1 relative">
        <div ref={mapContainer} className="w-full h-full" />
      </div>

      {/* Time control panel - bottom */}
      <div className="bg-panel px-6 py-4 flex items-center gap-6 flex-wrap">
        {/* Date selection */}
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-text-primary text-sm font-medium">
            From:
            <input
              type="date"
              value={startDate}
              max={todayString}
              onChange={(e) => setStartDate(e.target.value)}
              className="px-3 py-2 border border-gray-600 rounded-md text-sm text-text-primary bg-ocean focus:outline-none focus:border-btn focus:ring-2 focus:ring-btn/30"
            />
          </label>
          <label className="flex items-center gap-2 text-text-primary text-sm font-medium">
            To:
            <input
              type="date"
              value={endDate}
              max={todayString}
              onChange={(e) => setEndDate(e.target.value)}
              className="px-3 py-2 border border-gray-600 rounded-md text-sm text-text-primary bg-ocean focus:outline-none focus:border-btn focus:ring-2 focus:ring-btn/30"
            />
          </label>
          <button
            onClick={loadTimeRangeData}
            disabled={isTimeRangeLoading}
            className="px-5 py-2 bg-btn text-text-primary border-none rounded-md text-sm font-medium cursor-pointer transition-colors hover:bg-btn-hover disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isTimeRangeLoading ? "Loading..." : "Load"}
          </button>
        </div>

        {/* Separator */}
        {availableDays.length > 0 && (
          <div className="w-px h-8 bg-gray-600" />
        )}

        {/* Slider and simulation */}
        {availableDays.length > 0 && (
          <div className="flex items-center gap-4 flex-1">
            <button
              onClick={toggleSimulation}
              className={`px-4 py-2 border-none rounded-md text-sm font-medium cursor-pointer transition-colors ${
                isSimulationRunning
                  ? "bg-point hover:opacity-80 text-text-primary"
                  : "bg-btn hover:bg-btn-hover text-text-primary"
              }`}
            >
              {isSimulationRunning ? "Stop" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={availableDays.length - 1}
              value={selectedDayIndex}
              onChange={(e) => setSelectedHourIndex(Number(e.target.value))}
              className="flex-1 h-2 bg-ocean rounded cursor-pointer accent-btn"
            />
            <span className="min-w-[160px] text-center text-sm font-semibold text-text-primary bg-ocean px-3 py-2 rounded-md">
              {availableDays[selectedDayIndex]}
            </span>
          </div>
        )}
      </div>

      {/* Dialog */}
      {isDialogOpen && (
        <div
          onClick={closeDialog}
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-[9999]"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="bg-panel rounded-xl p-8 min-w-[500px] relative shadow-2xl border border-ocean"
          >
            <button
              onClick={closeDialog}
              className="absolute top-4 right-4 bg-transparent border-none text-2xl cursor-pointer text-text-primary hover:text-btn transition-colors"
            >
              &times;
            </button>
            <h2 className="m-0 mb-6 text-text-primary text-xl text-center font-semibold">
              {selectedPoint ? `${selectedPoint.lat.toFixed(4)}, ${selectedPoint.lng.toFixed(4)}` : "Loading..."}
            </h2>
            <div className="w-[450px] h-[320px] rounded-lg overflow-hidden flex items-center justify-center bg-ocean relative">
              {isLoading ? (
                <span className="text-text-primary">Loading...</span>
              ) : (
                <img
                  src={isUpscaled ? "/upscale.png" : "/downscale.png"}
                  alt="Ship image"
                  className="w-full h-full object-cover"
                />
              )}
              {/* Upscaling overlay */}
              {isUpscaling && (
                <div className="absolute inset-0 bg-ocean/80 flex items-center justify-center">
                  <span className="text-text-primary text-lg">Upscaling...</span>
                </div>
              )}
            </div>
            {/* Upscale button */}
            <button
              onClick={handleUpscale}
              disabled={isUpscaling || isUpscaled || isLoading}
              className="mt-4 w-full py-3 bg-btn text-text-primary rounded-md font-medium hover:bg-btn-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isUpscaled ? "Upscaled ✓" : isUpscaling ? "Upscaling..." : "Upscale Image"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
