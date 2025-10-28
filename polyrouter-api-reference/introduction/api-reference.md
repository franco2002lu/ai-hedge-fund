# API Reference

> Interactive API documentation for all PolyRouter endpoints

<Warning>
  **Open Beta:** PolyRouter is currently in open beta. All features are free during this period. Rate limit: 10 req/sec for all users.
</Warning>

## Interactive API Testing

This documentation allows you to explore and test all endpoints directly in your browser with:

* **Live Testing** - Try requests with your API key
* **Code Examples** - Copy-paste ready code
* **Response Schemas** - Detailed structure documentation

<Info>
  Click on any endpoint below to access the interactive playground with live testing.
</Info>

## Authentication

All endpoints require an API key in the `X-API-Key` header:

```bash  theme={null}
curl "https://api.polyrouter.io/functions/v1/markets-v2" \
  -H "X-API-Key: YOUR_API_KEY"
```

<Note>
  [Get your free API key](https://polyrouter.io) - takes 30 seconds.
</Note>

**Base URL:** `https://api.polyrouter.io/functions/v1`

## Sports V1 API

Unified sports betting with standardized game IDs across 5+ platforms.

<CardGroup cols={2}>
  <Card title="League Info" icon="info-circle" href="#get-league-info-v1">
    Team mappings and league metadata
  </Card>

  <Card title="List Games" icon="list" href="#get-list-games-v1">
    Find available games with IDs
  </Card>

  <Card title="Game Markets" icon="football" href="#get-games-v1-game-id">
    Betting markets from 5+ platforms
  </Card>

  <Card title="Awards" icon="trophy" href="#get-awards-v1">
    MVP, Super Bowl odds
  </Card>
</CardGroup>

## Markets V2 API

Real-time prediction market data with direct platform integration.

<CardGroup cols={2}>
  <Card title="Markets" icon="chart-line" href="#get-markets-v2">
    Browse and filter markets
  </Card>

  <Card title="Events" icon="calendar" href="#get-events-v2">
    Prediction market events
  </Card>

  <Card title="Series" icon="layer-group" href="#get-series-v2">
    Recurring market series
  </Card>

  <Card title="Search" icon="magnifying-glass" href="#get-search-v2">
    Multi-platform search
  </Card>

  <Card title="Price History" icon="chart-area" href="#get-price-history-v2">
    Historical OHLC data
  </Card>

  <Card title="Platforms" icon="server" href="#get-platforms-v2">
    Platform information
  </Card>
</CardGroup>

## Need Help?

<CardGroup cols={2}>
  <Card title="Join Discord" icon="discord" href="https://discord.gg/fyagg92CVM">
    Community support
  </Card>

  <Card title="Follow on X" icon="x-twitter" href="https://x.com/polyrouter">
    Latest updates
  </Card>
</CardGroup>
