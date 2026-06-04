import { NextRequest, NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function proxy(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const url = `${API_BASE}/${path.join("/")}`;
  const res = await fetch(url, { method: req.method, headers: req.headers, body: req.method !== "GET" ? req.body : undefined, duplex: "half" } as RequestInit);
  return new NextResponse(res.body, { status: res.status, headers: res.headers });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const DELETE = proxy;