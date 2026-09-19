import { useCallback, useState } from "react";

const KEY = "hrte.admin_token";

/** Optional gate -- see app/api/routes_admin.py::_require_admin. When
 * HRTE_ADMIN_TOKEN isn't set server-side (the default, e.g. local dev), this
 * is simply unused and every admin call succeeds without it.
 */
export function useAdminToken() {
  const [adminToken, setAdminTokenState] = useState<string>(() => sessionStorage.getItem(KEY) || "");

  const setAdminToken = useCallback((value: string) => {
    setAdminTokenState(value);
    if (value) sessionStorage.setItem(KEY, value);
    else sessionStorage.removeItem(KEY);
  }, []);

  return { adminToken: adminToken || undefined, setAdminToken };
}
