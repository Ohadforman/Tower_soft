import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import AppKit

let args = CommandLine.arguments
if args.count < 4 {
    fputs("usage: swift render_manual_page.swift <pdf-path> <page-number> <output-path>\n", stderr)
    exit(2)
}

let pdfURL = URL(fileURLWithPath: args[1])
let pageIndex = max(0, (Int(args[2]) ?? 1) - 1)
let outURL = URL(fileURLWithPath: args[3])

guard let document = CGPDFDocument(pdfURL as CFURL), let page = document.page(at: pageIndex + 1) else {
    fputs("failed to open pdf or page\n", stderr)
    exit(1)
}

let bounds = page.getBoxRect(.mediaBox)
let scale: CGFloat = 3.0
let pixelWidth = max(1, Int(bounds.width * scale))
let pixelHeight = max(1, Int(bounds.height * scale))

guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
    fputs("failed to create color space\n", stderr)
    exit(1)
}

guard let context = CGContext(
    data: nil,
    width: pixelWidth,
    height: pixelHeight,
    bitsPerComponent: 8,
    bytesPerRow: 0,
    space: colorSpace,
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
) else {
    fputs("failed to create bitmap context\n", stderr)
    exit(1)
}

context.setFillColor(NSColor.white.cgColor)
context.fill(CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))
context.interpolationQuality = .high
context.setAllowsAntialiasing(true)
context.setShouldAntialias(true)

let drawRect = CGRect(x: 0, y: 0, width: CGFloat(pixelWidth), height: CGFloat(pixelHeight))
let transform = page.getDrawingTransform(.mediaBox, rect: drawRect, rotate: 0, preserveAspectRatio: true)
context.concatenate(transform)
context.drawPDFPage(page)

guard let image = context.makeImage() else {
    fputs("failed to create image\n", stderr)
    exit(1)
}

func croppedContentImage(from image: CGImage, context: CGContext) -> CGImage {
    guard
        let data = context.data
    else {
        return image
    }

    let providerData = data.bindMemory(to: UInt8.self, capacity: context.height * context.bytesPerRow)

    let width = context.width
    let height = context.height
    let bytesPerRow = context.bytesPerRow
    let whiteThreshold: UInt8 = 246
    let alphaThreshold: UInt8 = 8

    var minX = width
    var minY = height
    var maxX = -1
    var maxY = -1

    for y in 0..<height {
        let row = providerData.advanced(by: y * bytesPerRow)
        for x in 0..<width {
            let pixel = row.advanced(by: x * 4)
            let r = pixel[0]
            let g = pixel[1]
            let b = pixel[2]
            let a = pixel[3]
            let isInk = a > alphaThreshold && (r < whiteThreshold || g < whiteThreshold || b < whiteThreshold)
            if isInk {
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
            }
        }
    }

    guard maxX >= minX, maxY >= minY else {
        return image
    }

    let inkWidth = maxX - minX + 1
    let inkHeight = maxY - minY + 1
    let fullArea = width * height
    let inkArea = inkWidth * inkHeight

    guard inkArea > 0, inkArea < fullArea else {
        return image
    }

    let cropRatio = Double(inkArea) / Double(fullArea)
    let shouldCrop = cropRatio < 0.82 || minX > width / 20 || minY > height / 20
    guard shouldCrop else {
        return image
    }

    let margin = max(18, min(width, height) / 40)
    let cropRect = CGRect(
        x: max(0, minX - margin),
        y: max(0, minY - margin),
        width: min(width - max(0, minX - margin), inkWidth + (margin * 2)),
        height: min(height - max(0, minY - margin), inkHeight + (margin * 2))
    ).integral

    guard cropRect.width > 0, cropRect.height > 0 else {
        return image
    }

    return image.cropping(to: cropRect) ?? image
}

let croppedImage = croppedContentImage(from: image, context: context)
let bitmap = NSBitmapImageRep(cgImage: croppedImage)
guard let pngData = bitmap.representation(using: .png, properties: [:]) else {
    fputs("failed png encode\n", stderr)
    exit(1)
}

do {
    try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try pngData.write(to: outURL)
    print(outURL.path)
} catch {
    fputs("failed to write png: \(error)\n", stderr)
    exit(1)
}
