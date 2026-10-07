import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";

export default defineConfig([
  ...nextVitals,
  // Existing localStorage hydration intentionally sets state inside effects.
  { rules: { "react-hooks/set-state-in-effect": "off" } },
  globalIgnores([".next/**", "node_modules/**"]),
]);
