import { Courier_Prime } from "next/font/google";
import "./globals.css";

const courierPrime = Courier_Prime({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-courier-prime",
});

export const metadata = {
  title: "ShipTracker",
  description: "Ship tracking application",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${courierPrime.className} antialiased`}>
        {children}
      </body>
    </html>
  );
}
