# Welcome to PolyRouter

> The unified API for prediction markets

<img className="block dark:hidden" src="https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=795a66b63926771da4cb4ab3db8f9bec" alt="PolyRouter Hero Light" data-og-width="1872" width="1872" data-og-height="375" height="375" data-path="images/logo_w_color.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=280&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=a736049a0e7784e40111c1428fa1f6d6 280w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=560&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=d3773cdab28ecadd7f372de05520f009 560w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=840&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=595c1c025edd779a3c19ed10a550bce6 840w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=1100&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=6f247b54c384913c20878d1d549f5432 1100w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=1650&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=4c729fb6911742a3f9f700e18248f1a2 1650w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=2500&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=818a8e1b1d34c0d95eef8f312166da9b 2500w" />

<img className="hidden dark:block" src="https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=795a66b63926771da4cb4ab3db8f9bec" alt="PolyRouter Hero Dark" data-og-width="1872" width="1872" data-og-height="375" height="375" data-path="images/logo_w_color.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=280&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=a736049a0e7784e40111c1428fa1f6d6 280w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=560&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=d3773cdab28ecadd7f372de05520f009 560w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=840&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=595c1c025edd779a3c19ed10a550bce6 840w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=1100&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=6f247b54c384913c20878d1d549f5432 1100w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=1650&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=4c729fb6911742a3f9f700e18248f1a2 1650w, https://mintcdn.com/polyrouter/u3ViWloqqpWw1oA4/images/logo_w_color.png?w=2500&fit=max&auto=format&n=u3ViWloqqpWw1oA4&q=85&s=818a8e1b1d34c0d95eef8f312166da9b 2500w" />

## What is PolyRouter?

<Warning>
  **Open Beta:** PolyRouter is currently in open beta. All features are free with 10 req/sec rate limits. We're actively improving the platform based on your feedback!
</Warning>

PolyRouter provides **unified API access to 7 prediction market platforms**: Polymarket, Kalshi, Manifold Markets, Limitless, ProphetX, Novig, and SX.bet. Get real-time market data, sports betting odds, and historical prices through one simple API.

<CardGroup cols={3}>
  <Card title="Unified Access" icon="layer-group" iconType="duotone">
    One API for 7 platforms
  </Card>

  <Card title="Real-Time Data" icon="bolt" iconType="duotone">
    Sub-second response times
  </Card>

  <Card title="Sports Betting" icon="football" iconType="duotone">
    NFL markets with standardized IDs
  </Card>
</CardGroup>

## Quick Start

<Steps>
  <Step title="Get Your API Key">
    [Sign up for free](https://polyrouter.io) and get your API key instantly.
  </Step>

  <Step title="Make Your First Request">
    Use the examples below to start fetching data.
  </Step>

  <Step title="Explore the API">
    Check out the [interactive API docs](/api-reference/introduction).
  </Step>
</Steps>

## Examples

### Fetch Markets

<CodeGroup>
  ```bash cURL theme={null}
  curl "https://api.polyrouter.io/functions/v1/markets-v2?platform=polymarket&limit=5" \
    -H "X-API-Key: YOUR_API_KEY"
  ```

  ```javascript JavaScript theme={null}
  const response = await fetch(
    'https://api.polyrouter.io/functions/v1/markets-v2?platform=polymarket&limit=5',
    { headers: { 'X-API-Key': 'YOUR_API_KEY' } }
  );

  const data = await response.json();
  data.markets.forEach(market => {
    console.log(`${market.title}: ${(market.current_prices.yes.price * 100).toFixed(1)}%`);
  });
  ```
</CodeGroup>

<Accordion title="Response Example">
  ```json  theme={null}
  {
    "markets": [
      {
        "id": "516710",
        "platform": "polymarket",
        "title": "US recession in 2025?",
        "current_prices": {
          "yes": { "price": 0.065 },
          "no": { "price": 0.935 }
        },
        "volume_24h": 14627.93,
        "status": "open"
      }
    ]
  }
  ```
</Accordion>

### Get NFL Odds

```bash  theme={null}
# Find today's games
curl "https://api.polyrouter.io/functions/v1/list-games-v1?league=nfl&limit=5" \
  -H "X-API-Key: YOUR_API_KEY"

# Get odds for a specific game
curl "https://api.polyrouter.io/functions/v1/games-v1/BUFvKC20251020" \
  -H "X-API-Key: YOUR_API_KEY"
```

<Info>
  **Game IDs** follow the format `{AwayTeam}v{HomeTeam}{YYYYMMDD}`. Use `/list-games-v1` to discover available games.
</Info>

### Query Price History

```bash  theme={null}
curl "https://api.polyrouter.io/functions/v1/price-history-v2?market_ids=516710&interval=1h&limit=24" \
  -H "X-API-Key: YOUR_API_KEY"
```

## Supported Platforms

<AccordionGroup>
  <Accordion icon="chart-simple" title="Polymarket">
    Largest decentralized prediction market • CLOB order books • Event/series hierarchy
  </Accordion>

  <Accordion icon="building-columns" title="Kalshi">
    US-regulated with CFTC oversight • Binary markets • Settlement timers
  </Accordion>

  <Accordion icon="infinity" title="Limitless">
    Community-driven markets • Multiple collateral tokens • Low barriers
  </Accordion>

  <Accordion icon="chart-mixed" title="ProphetX">
    Tournament-based sports • 30-50 spread/total options per game
  </Accordion>

  <Accordion icon="dice" title="Novig">
    Most comprehensive player props • 200+ props per NFL game • Deep liquidity
  </Accordion>

  <Accordion icon="circle-nodes" title="SX.bet">
    Decentralized exchange • On-chain order books • Full transparency
  </Accordion>

  <Accordion icon="sparkles" title="Manifold Markets">
    Play-money prediction markets • Community predictions • Free to participate
  </Accordion>
</AccordionGroup>

## API Features

<CardGroup cols={2}>
  <Card title="Markets V2" icon="chart-line" href="/api-reference/introduction">
    Real-time markets, events, and series
  </Card>

  <Card title="Sports V1" icon="football" href="/api-reference/introduction">
    NFL game markets and player props
  </Card>

  <Card title="Price History" icon="chart-area" href="/api-reference/introduction">
    OHLC data: 1m, 5m, 1h, 4h, 1d
  </Card>

  <Card title="Search" icon="magnifying-glass" href="/api-reference/introduction">
    Multi-platform search
  </Card>
</CardGroup>

## Rate Limits

<Note>
  **Open Beta:** All users currently have **10 requests per second** while we optimize our infrastructure. No paid plans yet — stay tuned for tiered pricing coming soon!
</Note>

## Resources

<CardGroup cols={2}>
  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Interactive documentation
  </Card>

  <Card title="Get API Key" icon="key" href="https://polyrouter.io">
    Sign up free
  </Card>

  <Card title="Discord Community" icon="discord" href="https://discord.gg/fyagg92CVM">
    Get help and support
  </Card>

  <Card title="Follow on X" icon="x-twitter" href="https://x.com/polyrouter">
    Latest updates
  </Card>
</CardGroup>
