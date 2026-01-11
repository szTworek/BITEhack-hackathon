class CoordConverter:
    def __init__(self, img_width, img_height, corners):
        self.W = img_width
        self.H = img_height
        self.lon_left = corners['top_left'][0]
        self.lat_top = corners['top_left'][1]
        self.lon_right = corners['bottom_right'][0]
        self.lat_bottom = corners['bottom_right'][1]
        self.delta_lon = self.lon_right - self.lon_left
        self.delta_lat = self.lat_top - self.lat_bottom

    def pixel_to_geo(self, x, y):
        lon = self.lon_left + (x / self.W) * self.delta_lon
        lat = self.lat_top - (y / self.H) * self.delta_lat
        return round(lon, 6), round(lat, 6)

    def get_bbox_center_geo(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        return self.pixel_to_geo(x_center, y_center)