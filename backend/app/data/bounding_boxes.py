"""
Bounding boxes for satellite image acquisition.
Each area defines a geographic region for monitoring ship traffic.
Format: [min_lon, min_lat, max_lon, max_lat]
"""

AREAS = {
"algeciras": {"bbox": [-5.48, 36.10, -5.38, 36.20], "desc": "Port Algeciras, Hiszpania"},
"busan": {"bbox": [129.00, 35.05, 129.15, 35.15], "desc": "Busan, Korea Południowa"},
"gibraltar": {"bbox": [-5.45, 36.05, -5.25, 36.20], "desc": "Skała Gibraltarska"},
"jebel_ali": {"bbox": [54.95, 24.95, 55.15, 25.10], "desc": "Dubaj, Jebel Ali"},
"los_angeles": {"bbox": [-118.30, 33.70, -118.20, 33.80], "desc": "Port Los Angeles"},
"panama_atlantic": {"bbox": [-79.95, 9.30, -79.85, 9.45], "desc": "Wejście do Kanału Panamskiego (Atlantyk)"},
"panama_pacific": {"bbox": [-79.58, 8.85, -79.48, 9.00], "desc": "Podejście do Kanału Panamskiego (Pacyfik)"},
"dover_strait": {"bbox": [1.25, 51.10, 1.45, 51.25], "desc": "Cieśnina Dover, Wielka Brytania"},
"valparaiso_chile": {"bbox": [-71.66, -33.05, -71.56, -32.95], "desc": "Valparaíso, Chile - główny port kontenerowy"},
"port_of_sydney": {"bbox": [151.18, -33.88, 151.32, -33.78], "desc": "Port Sydney, Australia"},
"vitoria_espirito": {"bbox": [-40.35, -20.35, -40.25, -20.25], "desc": "Vitória, Espírito Santo - port rudy żelaza"},
"manta_ecuador": {"bbox": [-80.85, -1.00, -80.68, -0.83], "desc": "Manta, Ekwador - 'stolica tuńczyka', bardzo jasny punkt"},
"zhoushan_china": {"bbox": [122.10, 29.85, 122.27, 30.02], "desc": "Archipelag Zhoushan, Chiny - prawdopodobnie największe skupisko łodzi na świecie"},
"fuzhou_strait_china": {"bbox": [119.80, 25.80, 119.97, 25.97], "desc": "Cieśnina Tajwańska (strona chińska) - ekstremalnie gęsto"},
"haiphong_vietnam": {"bbox": [106.85, 20.70, 107.02, 20.87], "desc": "Zatoka Tonkińska, Wietnam - bardzo jasna strefa rybacka"},
"agadir_morocco": {"bbox": [-9.80, 30.35, -9.63, 30.52], "desc": "Agadir, Maroko - intensywna strefa rybacka (ciepły kolor)"},
"indian_hot_35": {"bbox": [55.00, -20.00, 55.17, -19.83], "desc": "Na wschód od Madagaskaru"},
"SA_coastal_01": {"bbox": [-81.20, -5.10, -81.03, -4.93], "desc": "Paita, Peru - ekstremalne zagęszczenie przy porcie"},
"SA_coastal_03": {"bbox": [-77.20, -12.10, -77.03, -11.93], "desc": "Callao/Lima, Peru - główne kotwicowisko"},
"SA_coastal_05": {"bbox": [-80.80, -1.00, -80.63, -0.83], "desc": "Manta, Ekwador - zagłębie tuńczykowe"},
"SA_coastal_07": {"bbox": [-71.65, -33.00, -71.48, -32.83], "desc": "Valparaíso, Chile - duży ruch kontenerowy"},
"SA_coastal_08": {"bbox": [-73.15, -36.70, -72.98, -36.53], "desc": "Concepción, Chile - zatoka San Vicente"},
"SA_coastal_11": {"bbox": [-38.55, -12.95, -38.38, -12.78], "desc": "Salvador, Brazylia - ruch przybrzeżny"},
"SA_coastal_14": {"bbox": [-58.35, -34.65, -58.18, -34.48], "desc": "Buenos Aires, Argentyna - Rio de la Plata"},
"SA_coastal_25": {"bbox": [-44.40, -2.50, -44.23, -2.33], "desc": "São Luís, Brazylia - terminale rudy żelaza"},
"AF_coastal_26": {"bbox": [-17.45, 14.65, -17.28, 14.82], "desc": "Dakar, Senegal - kluczowy port Afryki Zach."},
"AF_coastal_31": {"bbox": [0.00, 5.60, 0.17, 5.77], "desc": "Tema, Ghana - duży port kontenerowy"},
"AF_coastal_40": {"bbox": [31.05, -29.85, 31.22, -29.68], "desc": "Durban, RPA - największy port kontenerowy Afryki"},
"AF_coastal_44": {"bbox": [32.30, 31.20, 32.47, 31.37], "desc": "Damietta, Egipt - Delta Nilu"},
}
