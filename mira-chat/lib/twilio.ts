const TWILIO_API = "https://api.twilio.com/2010-04-01";

const E164_REGEX = /^\+[1-9]\d{1,14}$/;

function getConfig() {
  const accountSid = process.env.TWILIO_ACCOUNT_SID;
  const authToken = process.env.TWILIO_AUTH_TOKEN;
  const from = process.env.TWILIO_PHONE_NUMBER;
  const defaultTo = process.env.TWILIO_ALERT_TO;
  if (!accountSid || !authToken || !from) return null;
  return { accountSid, authToken, from, defaultTo };
}

export function isE164(phone: string): boolean {
  return E164_REGEX.test(phone);
}

export async function sendSms(message: string, toOverride?: string): Promise<boolean> {
  const config = getConfig();
  if (!config) {
    console.warn("[Twilio] SMS skipped: missing env vars");
    return false;
  }

  const to = toOverride || config.defaultTo;
  if (!to) {
    console.warn("[Twilio] SMS skipped: no target phone number");
    return false;
  }

  if (!isE164(to)) {
    console.error("[Twilio] Invalid E.164 phone number:", to);
    return false;
  }

  const auth = Buffer.from(`${config.accountSid}:${config.authToken}`).toString("base64");
  const form = new URLSearchParams({
    To: to,
    From: config.from,
    Body: message,
  });

  try {
    const res = await fetch(`${TWILIO_API}/Accounts/${config.accountSid}/Messages.json`, {
      method: "POST",
      headers: {
        Authorization: `Basic ${auth}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: form.toString(),
    });
    if (!res.ok) {
      console.error("[Twilio] SMS failed:", res.status, await res.text());
      return false;
    }
    console.log("[Twilio] SMS sent to", to);
    return true;
  } catch (e) {
    console.error("[Twilio] SMS error:", e);
    return false;
  }
}
