interface ActionBannerProps {
  text: string;
  bannerKey: number;
  isAi: boolean;
}

export default function ActionBanner({ text, bannerKey, isAi }: ActionBannerProps) {
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
      <div
        key={bannerKey}
        className={`px-5 py-2.5 rounded-full bg-zinc-900/80 backdrop-blur-sm text-sm font-medium tracking-wide animate-banner-show ${
          isAi ? "text-zinc-300" : "text-emerald-400"
        }`}
      >
        {text}
      </div>
    </div>
  );
}
