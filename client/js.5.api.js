(global => {
  kubin.requestApi = async (apiUrl, body) => {
    const origin = global.location.origin
    const url = `${origin}/${apiUrl}`
    const res = await fetch(url, {
      "headers": {
        "accept": "*/*",
        "content-type": "application/json"
      },
      "body": body || JSON.stringify({ data: [[], null], event_data: null, fn_index: 0, session_hash: kubin.client_id }),
      "method": "POST",
      "mode": "cors",
      "credentials": "include"
    })

    const json = await res.json()
    return json
  }
})(window)
