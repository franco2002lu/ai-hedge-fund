# Get platform information

> Retrieve comprehensive information about all supported platforms, their capabilities, features, and real-time health status. This endpoint provides platform discovery, feature detection, ID format specifications, and health monitoring for both market and sports platforms.

## OpenAPI

````yaml openapi.json get /platforms
paths:
  path: /platforms
  method: get
  servers:
    - url: https://api.polyrouter.io/functions/v1
      description: Production server
  request:
    security:
      - title: ApiKeyAuth
        parameters:
          query: {}
          header:
            X-API-Key:
              type: apiKey
              description: API key for authentication
          cookie: {}
    parameters:
      path: {}
      query: {}
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              markets:
                allOf:
                  - type: object
                    properties:
                      platforms:
                        type: array
                        items:
                          $ref: '#/components/schemas/MarketPlatform'
                      health:
                        type: object
                        additionalProperties:
                          type: string
                          enum:
                            - healthy
                            - degraded
                            - unavailable
                        description: Health status for each market platform
                      total_platforms:
                        type: number
                        description: Total number of market platforms
              sports:
                allOf:
                  - type: object
                    properties:
                      platforms:
                        type: array
                        items:
                          $ref: '#/components/schemas/SportsPlatform'
                      health:
                        type: object
                        additionalProperties:
                          type: string
                          enum:
                            - healthy
                            - degraded
                            - unavailable
                        description: Health status for each sports platform
                      total_platforms:
                        type: number
                        description: Total number of sports platforms
              meta:
                allOf:
                  - type: object
                    properties:
                      request_time:
                        type: number
                        description: Request processing time (ms)
                      timestamp:
                        type: string
                        format: date-time
                        description: Response timestamp
                      version:
                        type: string
                        description: API version
        examples:
          example:
            value:
              markets:
                platforms:
                  - platform: polymarket
                    display_name: Polymarket
                    endpoints:
                      markets: true
                      events: true
                      series: true
                      search: true
                      price_history: true
                    features:
                      status_filtering: true
                      date_filtering: true
                      pagination_type: offset
                      market_types:
                        - <string>
                    id_format:
                      description: <string>
                      example: <string>
                      pattern: <string>
                    base_url: https://gamma-api.polymarket.com
                    rate_limit: 100 req/min
                health: {}
                total_platforms: 123
              sports:
                platforms:
                  - platform: polymarket
                    display_name: Polymarket Sports
                    endpoints:
                      awards: true
                      games: true
                      list_games: true
                      league_info: true
                    features:
                      odds_formats:
                        - american
                      supported_leagues:
                        - <string>
                      market_types:
                        - <string>
                    rate_limit: 100 req/min
                health: {}
                total_platforms: 123
              meta:
                request_time: 123
                timestamp: '2023-11-07T05:31:56Z'
                version: <string>
        description: Platform information retrieved successfully
    '401':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
        description: Authentication error response
    '500':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Internal server error response
  deprecated: false
  type: path
components:
  schemas:
    Meta:
      type: object
      required:
        - request_time
        - cache_hit
        - data_freshness
      properties:
        request_time:
          type: number
          description: Request processing time (ms)
        cache_hit:
          type: boolean
          description: Whether response was served from cache
        data_freshness:
          type: string
          format: date-time
          description: Data freshness timestamp
    Error:
      type: object
      required:
        - code
        - message
        - timestamp
      properties:
        code:
          type: string
          description: Error code
          example: VALIDATION_ERROR
        message:
          type: string
          description: Error message
        details:
          type: object
          description: Additional error details
        timestamp:
          type: string
          format: date-time
          description: Error timestamp
    MarketPlatform:
      type: object
      required:
        - platform
        - display_name
        - endpoints
        - features
        - id_format
        - base_url
        - rate_limit
      properties:
        platform:
          type: string
          description: Platform identifier (lowercase)
          example: polymarket
        display_name:
          type: string
          description: Human-readable platform name
          example: Polymarket
        endpoints:
          type: object
          properties:
            markets:
              type: boolean
              description: Supports Markets-v2 endpoint
            events:
              type: boolean
              description: Supports Events-v2 endpoint
            series:
              type: boolean
              description: Supports Series-v2 endpoint
            search:
              type: boolean
              description: Supports Search-v2 endpoint
            price_history:
              type: boolean
              description: Supports Price History-v2 endpoint
        features:
          type: object
          properties:
            status_filtering:
              type: boolean
              description: Supports filtering by market status
            date_filtering:
              type: boolean
              description: Supports filtering by date ranges
            pagination_type:
              type: string
              enum:
                - offset
                - cursor
              description: Pagination method
            market_types:
              type: array
              items:
                type: string
              description: Supported market types
        id_format:
          type: object
          properties:
            description:
              type: string
              description: Human-readable description of ID format
            example:
              type: string
              description: Example market ID
            pattern:
              type: string
              description: Regular expression pattern for validation
        base_url:
          type: string
          description: Platform's base API URL
          example: https://gamma-api.polymarket.com
        rate_limit:
          type: string
          description: Rate limit information
          example: 100 req/min
    SportsPlatform:
      type: object
      required:
        - platform
        - display_name
        - endpoints
        - features
        - rate_limit
      properties:
        platform:
          type: string
          description: Platform identifier (lowercase)
          example: polymarket
        display_name:
          type: string
          description: Human-readable platform name
          example: Polymarket Sports
        endpoints:
          type: object
          properties:
            awards:
              type: boolean
              description: Supports Awards-v1 endpoint
            games:
              type: boolean
              description: Supports Games-v1 endpoint
            list_games:
              type: boolean
              description: Supports List-Games-v1 endpoint
            league_info:
              type: boolean
              description: Supports League-Info-v1 endpoint
        features:
          type: object
          properties:
            odds_formats:
              type: array
              items:
                type: string
                enum:
                  - american
                  - decimal
                  - probability
              description: Supported odds formats
            supported_leagues:
              type: array
              items:
                type: string
              description: Supported sports leagues
            market_types:
              type: array
              items:
                type: string
              description: Supported market types
        rate_limit:
          type: string
          description: Rate limit information
          example: 100 req/min

````